#include "cs2mv/gc.h"

#include "cs2mv/protobuf.h"

namespace cs2mv {
namespace {

// Reads a repeated int32 field that may arrive either packed or unpacked.
void ReadRepeatedInt32(pb::Reader* r, std::vector<int>* out) {
  if (r->wire_type() == pb::kLengthDelimited) {
    const pb::Slice s = r->ReadBytes();
    pb::Reader packed(s);
    while (packed.remaining() > 0) {
      const int v = static_cast<int>(packed.ReadVarint());
      if (!packed.ok()) break;
      out->push_back(v);
    }
    return;
  }
  out->push_back(r->ReadInt32());
}

void ReadRepeatedUInt32(pb::Reader* r, std::vector<std::uint32_t>* out) {
  if (r->wire_type() == pb::kLengthDelimited) {
    const pb::Slice s = r->ReadBytes();
    pb::Reader packed(s);
    while (packed.remaining() > 0) {
      const std::uint32_t v = static_cast<std::uint32_t>(packed.ReadVarint());
      if (!packed.ok()) break;
      out->push_back(v);
    }
    return;
  }
  out->push_back(r->ReadUInt32());
}

bool LooksLikeUrl(const std::string& s) {
  return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

// CMsgGCCStrike15_v2_MatchmakingGC2ServerReserve { account_ids = 1 }
void ParseReservation(const pb::Slice& s, GcRoundStats* out) {
  pb::Reader r(s);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (field == 1) ReadRepeatedUInt32(&r, &out->account_ids);
  }
}

// CMsgGCCStrike15_v2_MatchmakingServerRoundStats
void ParseRoundStats(const pb::Slice& s, GcRoundStats* out) {
  pb::Reader r(s);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    switch (field) {
      case 1: out->reservation_id = r.ReadVarint(); break;
      case 2: ParseReservation(r.ReadBytes(), out); break;
      case 3: out->map = r.ReadString(); break;
      case 4: out->round = r.ReadInt32(); break;
      case 5: ReadRepeatedInt32(&r, &out->kills); break;
      case 6: ReadRepeatedInt32(&r, &out->assists); break;
      case 7: ReadRepeatedInt32(&r, &out->deaths); break;
      case 8: ReadRepeatedInt32(&r, &out->scores); break;
      case 10: out->round_result = r.ReadInt32(); break;
      case 11: out->match_result = r.ReadInt32(); break;
      case 12: ReadRepeatedInt32(&r, &out->team_scores); break;
      case 15: out->match_duration = r.ReadInt32(); break;
      case 16: ReadRepeatedInt32(&r, &out->enemy_kills); break;
      case 17: ReadRepeatedInt32(&r, &out->enemy_headshots); break;
      case 21: ReadRepeatedInt32(&r, &out->mvps); break;
      case 31: out->max_rounds = r.ReadUInt32(); break;
      default: break;
    }
  }
}

// WatchableMatchInfo { server_ip = 1, tv_port = 2, ... }
void ParseWatchable(const pb::Slice& s, GcMatchInfo* out) {
  pb::Reader r(s);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    switch (field) {
      case 1: out->server_ip = r.ReadUInt32(); break;
      case 2: out->tv_port = r.ReadUInt32(); break;
      default: break;
    }
  }
}

// CDataGCCStrike15_v2_MatchInfo
void ParseMatchInfo(const pb::Slice& s, GcMatchInfo* out) {
  pb::Reader r(s);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    switch (field) {
      case 1: out->match_id = r.ReadVarint(); break;
      case 2: out->match_time = r.ReadUInt32(); break;
      case 3: ParseWatchable(r.ReadBytes(), out); break;
      case 4:
      case 5: {
        GcRoundStats stats;
        ParseRoundStats(r.ReadBytes(), &stats);
        out->round_stats.push_back(std::move(stats));
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace

std::string GcMatchInfo::DemoUrl() const {
  for (auto it = round_stats.rbegin(); it != round_stats.rend(); ++it) {
    if (LooksLikeUrl(it->map)) return it->map;
  }
  return std::string();
}

std::vector<std::uint64_t> GcMatchInfo::SteamIds() const {
  std::vector<std::uint64_t> ids;
  for (auto it = round_stats.rbegin(); it != round_stats.rend(); ++it) {
    if (it->account_ids.empty()) continue;
    for (std::uint32_t account : it->account_ids) {
      if (account != 0) ids.push_back(kSteamId64Base + account);
    }
    break;
  }
  return ids;
}

std::string BuildMatchListRequest(const ShareCode& code) {
  pb::Writer w;
  w.AddVarint(1, code.match_id);
  w.AddVarint(2, code.outcome_id);
  w.AddVarint(3, code.token);
  return w.data();
}

bool ParseMatchList(const void* data, std::size_t size,
                    std::vector<GcMatchInfo>* out, std::string* error) {
  pb::Reader r(data, size);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (field != 4 || r.wire_type() != pb::kLengthDelimited) continue;
    GcMatchInfo info;
    ParseMatchInfo(r.ReadBytes(), &info);
    out->push_back(std::move(info));
  }
  if (!r.ok()) {
    if (error != nullptr) *error = "malformed CMsgGCCStrike15_v2_MatchList";
    return false;
  }
  return true;
}

std::string BuildMatchListReply(std::uint64_t match_id, std::uint32_t match_time,
                                const std::string& demo_url,
                                const std::vector<std::uint32_t>& account_ids) {
  pb::Writer reservation;
  for (std::uint32_t id : account_ids) reservation.AddVarint(1, id);

  pb::Writer stats;
  stats.AddVarint(1, match_id);
  stats.AddMessage(2, reservation);
  stats.AddString(3, demo_url);

  pb::Writer info;
  info.AddVarint(1, match_id);
  info.AddVarint(2, match_time);
  info.AddMessage(5, stats);

  pb::Writer list;
  list.AddMessage(4, info);
  return list.data();
}

}  // namespace cs2mv
