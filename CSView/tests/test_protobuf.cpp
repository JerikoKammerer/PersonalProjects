#include "cs2mv/protobuf.h"

#include <string>
#include <vector>

#include "cs2mv/gc.h"
#include "cs2mv/json.h"
#include "cs2mv/sharecode.h"
#include "test_util.h"

using cs2mv::JsonWriter;

TEST(ProtobufRoundTripsScalars) {
  cs2mv::pb::Writer w;
  w.AddVarint(1, 150);
  w.AddString(2, "hello");
  w.AddFixed64(3, 0x0123456789ABCDEFull);
  w.AddBool(4, true);
  w.AddFixed32(5, 0xDEADBEEFu);
  w.AddVarint(6, 18446744073709551615ull);

  cs2mv::pb::Reader r(w.data());
  std::uint32_t field = 0;
  int seen = 0;
  while (r.NextField(&field)) {
    switch (field) {
      case 1: CHECK_EQ(r.ReadVarint(), 150ull); ++seen; break;
      case 2: CHECK_EQ(r.ReadString(), std::string("hello")); ++seen; break;
      case 3: CHECK_EQ(r.ReadFixed64(), 0x0123456789ABCDEFull); ++seen; break;
      case 4: CHECK(r.ReadBool()); ++seen; break;
      case 5: CHECK_EQ(r.ReadFixed32(), 0xDEADBEEFu); ++seen; break;
      case 6: CHECK_EQ(r.ReadVarint(), 18446744073709551615ull); ++seen; break;
      default: break;
    }
  }
  CHECK(r.ok());
  CHECK_EQ(seen, 6);
}

TEST(ProtobufVarint150IsCanonical) {
  // The canonical example from the protobuf documentation.
  cs2mv::pb::Writer w;
  w.AddVarint(1, 150);
  CHECK_EQ(w.data(), std::string("\x08\x96\x01", 3));
}

TEST(ProtobufSkipsUnconsumedFields) {
  cs2mv::pb::Writer w;
  w.AddString(1, "skipped");
  w.AddVarint(2, 7);
  w.AddFixed64(3, 1);
  w.AddVarint(4, 42);

  cs2mv::pb::Reader r(w.data());
  std::uint32_t field = 0;
  std::uint64_t last = 0;
  while (r.NextField(&field)) {
    if (field == 4) last = r.ReadVarint();  // every earlier field goes unread
  }
  CHECK(r.ok());
  CHECK_EQ(last, 42ull);
}

TEST(ProtobufReportsTruncation) {
  cs2mv::pb::Writer w;
  w.AddString(1, "abcdefgh");
  std::string truncated = w.data();
  truncated.resize(truncated.size() - 3);

  cs2mv::pb::Reader r(truncated);
  std::uint32_t field = 0;
  while (r.NextField(&field)) r.ReadString();
  CHECK(!r.ok());
}

TEST(ProtobufNestedMessages) {
  cs2mv::pb::Writer inner;
  inner.AddString(2, "de_mirage");
  cs2mv::pb::Writer outer;
  outer.AddVarint(1, 9);
  outer.AddMessage(4, inner);

  cs2mv::pb::Reader r(outer.data());
  std::uint32_t field = 0;
  std::string map;
  while (r.NextField(&field)) {
    if (field != 4) continue;
    cs2mv::pb::Reader sub(r.ReadBytes());
    std::uint32_t sub_field = 0;
    while (sub.NextField(&sub_field)) {
      if (sub_field == 2) map = sub.ReadString();
    }
  }
  CHECK_EQ(map, std::string("de_mirage"));
}

TEST(GcRequestCarriesShareCodeFields) {
  cs2mv::ShareCode code;
  std::string error;
  CHECK(cs2mv::DecodeShareCode("CSGO-ENVEO-jHLV8-KFDmG-hxUWE-jfJ5O", &code, &error));

  const std::string request = cs2mv::BuildMatchListRequest(code);
  cs2mv::pb::Reader r(request);
  std::uint32_t field = 0;
  std::uint64_t match_id = 0, outcome_id = 0, token = 0;
  while (r.NextField(&field)) {
    if (field == 1) match_id = r.ReadVarint();
    if (field == 2) outcome_id = r.ReadVarint();
    if (field == 3) token = r.ReadVarint();
  }
  CHECK_EQ(match_id, code.match_id);
  CHECK_EQ(outcome_id, code.outcome_id);
  CHECK_EQ(token, static_cast<std::uint64_t>(code.token));
}

TEST(GcReplyYieldsDemoUrlAndPlayers) {
  const std::string url =
      "http://replay191.valve.net/730/003693258866644910353_1173271036.dem.bz2";
  const std::vector<std::uint32_t> accounts = {12345, 67890};
  const std::string reply =
      cs2mv::BuildMatchListReply(3253932207440199901ull, 1700000000u, url, accounts);

  std::vector<cs2mv::GcMatchInfo> matches;
  std::string error;
  CHECK(cs2mv::ParseMatchList(reply.data(), reply.size(), &matches, &error));
  CHECK_EQ(matches.size(), static_cast<std::size_t>(1));
  if (matches.empty()) return;

  CHECK_EQ(matches[0].match_id, 3253932207440199901ull);
  CHECK_EQ(matches[0].match_time, 1700000000u);
  CHECK_EQ(matches[0].DemoUrl(), url);

  const std::vector<std::uint64_t> ids = matches[0].SteamIds();
  CHECK_EQ(ids.size(), static_cast<std::size_t>(2));
  if (ids.size() == 2) {
    CHECK_EQ(ids[0], cs2mv::kSteamId64Base + 12345);
    CHECK_EQ(ids[1], cs2mv::kSteamId64Base + 67890);
  }
}

TEST(GcReplyWithoutUrlYieldsNothing) {
  // A match whose demo has expired reports a map name instead of a URL.
  const std::string reply =
      cs2mv::BuildMatchListReply(1, 0, "de_nuke", std::vector<std::uint32_t>());
  std::vector<cs2mv::GcMatchInfo> matches;
  std::string error;
  CHECK(cs2mv::ParseMatchList(reply.data(), reply.size(), &matches, &error));
  CHECK_EQ(matches.size(), static_cast<std::size_t>(1));
  if (!matches.empty()) CHECK_EQ(matches[0].DemoUrl(), std::string());
}

TEST(JsonWriterEscapesAndNests) {
  JsonWriter w;
  w.BeginObject();
  w.Field("name", std::string("say \"hi\"\n\\"));
  w.Field("count", 3);
  w.Field("ratio", 1.5);
  w.Field("flag", true);
  w.FieldId("bigId", 18446744073709551615ull);
  w.Key("list");
  w.BeginArray();
  w.Value(1);
  w.Value(2);
  w.EndArray();
  w.Key("empty");
  w.BeginObject();
  w.EndObject();
  w.EndObject();

  CHECK_EQ(w.str(),
           std::string("{\"name\":\"say \\\"hi\\\"\\n\\\\\",\"count\":3,"
                       "\"ratio\":1.5,\"flag\":true,"
                       "\"bigId\":\"18446744073709551615\","
                       "\"list\":[1,2],\"empty\":{}}"));
}

TEST(JsonWriterHandlesControlCharacters) {
  JsonWriter w;
  w.BeginObject();
  w.Field("c", std::string("\x01", 1));
  w.EndObject();
  CHECK_EQ(w.str(), std::string("{\"c\":\"\\u0001\"}"));
}
