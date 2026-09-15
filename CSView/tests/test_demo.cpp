// End to end test over tests/data/synthetic.dem, which tools/make_fixture.py
// builds as a real PBDEMS2 container: Snappy compressed frames, bit-packed
// Source 2 message framing, protobuf messages, a userinfo string table and
// legacy game events. Parsing it exercises every layer this project owns.
#include <string>

#include "cs2mv/demo.h"
#include "cs2mv/match.h"
#include "cs2mv/parser.h"
#include "test_util.h"

using cs2mv::DemoReader;
using cs2mv::Match;
using cs2mv::ParseDemo;
using cs2mv::ParseOptions;
using cs2mv::Player;

namespace {

const Player* Find(const Match& match, const std::string& name) {
  for (const Player& p : match.players) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

bool ParseFixture(const std::string& file, Match* match) {
  const std::string bytes = testing::ReadFileOrDie(file);
  DemoReader reader;
  std::string error;
  if (!reader.Init(bytes, &error)) {
    std::fprintf(stderr, "    demo init failed: %s\n", error.c_str());
    return false;
  }
  return ParseDemo(&reader, ParseOptions(), match, &error);
}

}  // namespace

TEST(DemoReaderRejectsNonDemo) {
  DemoReader reader;
  std::string error;
  CHECK(!reader.Init("HL2DEMO\0 some old csgo demo", &error));
  CHECK(!error.empty());
}

TEST(DemoReaderWalksFrames) {
  const std::string bytes = testing::ReadFileOrDie("synthetic.dem");
  DemoReader reader;
  std::string error;
  CHECK(reader.Init(bytes, &error));

  int frames = 0;
  int compressed = 0;
  bool saw_header = false;
  bool saw_string_tables = false;
  cs2mv::DemoFrame frame;
  while (reader.Next(&frame)) {
    ++frames;
    if (frame.compressed) ++compressed;
    if (frame.kind == cs2mv::kDemFileHeader) saw_header = true;
    if (frame.kind == cs2mv::kDemStringTables) saw_string_tables = true;
  }
  CHECK(!reader.failed());
  CHECK(frames > 5);
  CHECK(compressed > 0);   // the fixture Snappy-packs several frames
  CHECK(saw_header);
  CHECK(saw_string_tables);
}

TEST(DemoHeaderIsRead) {
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));
  CHECK_EQ(match.map_name, std::string("de_dust2"));
  CHECK_NEAR(match.tick_interval, 1.0 / 64.0, 1e-9);
  CHECK_EQ(match.build_number, 14000);
  CHECK(match.playback_ticks > 0);
}

TEST(DemoScoreboardMatchesFixture) {
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));

  CHECK_EQ(match.score_ct, 2);
  CHECK_EQ(match.score_t, 1);
  CHECK_EQ(match.rounds_played, 3);
  CHECK_EQ(match.players.size(), static_cast<std::size_t>(4));

  struct Expected {
    const char* name;
    int kills, deaths, assists, headshots, damage, mvps, entry_kills,
        enemies_flashed, utility_damage, team;
  };
  // These are the values tools/make_fixture.py reports when it builds the file.
  const Expected expected[] = {
      {"Ada", 3, 1, 0, 2, 340, 2, 2, 1, 40, cs2mv::kTeamCT},
      {"Bo", 1, 2, 0, 0, 100, 0, 0, 0, 0, cs2mv::kTeamCT},
      {"Cyd", 2, 2, 1, 0, 200, 0, 1, 0, 0, cs2mv::kTeamT},
      {"Dex", 1, 2, 0, 0, 100, 1, 0, 0, 0, cs2mv::kTeamT},
  };

  for (const Expected& e : expected) {
    const Player* p = Find(match, e.name);
    CHECK(p != nullptr);
    if (p == nullptr) continue;
    CHECK_EQ(p->kills, e.kills);
    CHECK_EQ(p->deaths, e.deaths);
    CHECK_EQ(p->assists, e.assists);
    CHECK_EQ(p->headshots, e.headshots);
    CHECK_EQ(p->damage, e.damage);
    CHECK_EQ(p->mvps, e.mvps);
    CHECK_EQ(p->entry_kills, e.entry_kills);
    CHECK_EQ(p->enemies_flashed, e.enemies_flashed);
    CHECK_EQ(p->utility_damage, e.utility_damage);
    CHECK_EQ(p->team, e.team);
    CHECK_EQ(p->rounds_played, 3);
    CHECK(p->steam_id != 0);
  }

  const Player* ada = Find(match, "Ada");
  if (ada != nullptr) {
    CHECK_NEAR(ada->adr(), 340.0 / 3.0, 0.01);
    CHECK_NEAR(ada->kd(), 3.0, 1e-9);
    CHECK_NEAR(ada->hs_percent(), 200.0 / 3.0, 0.01);
  }
}

TEST(DemoRoundsAreOrderedAndScored) {
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));
  CHECK_EQ(match.rounds.size(), static_cast<std::size_t>(3));
  if (match.rounds.size() != 3) return;

  CHECK_EQ(match.rounds[0].number, 1);
  CHECK_EQ(match.rounds[0].winner, cs2mv::kTeamCT);
  CHECK_EQ(match.rounds[0].score_ct, 1);
  CHECK_EQ(match.rounds[0].score_t, 0);
  CHECK_EQ(match.rounds[0].kills.size(), static_cast<std::size_t>(2));

  CHECK_EQ(match.rounds[1].winner, cs2mv::kTeamT);
  CHECK_EQ(match.rounds[1].score_ct, 1);
  CHECK_EQ(match.rounds[1].score_t, 1);
  CHECK(match.rounds[1].bomb_planted);

  CHECK_EQ(match.rounds[2].winner, cs2mv::kTeamCT);
  CHECK_EQ(match.rounds[2].score_ct, 2);
  CHECK_EQ(match.rounds[2].kills.size(), static_cast<std::size_t>(3));
  CHECK_EQ(match.rounds[2].reason_text, std::string("Terrorists eliminated"));
}

TEST(DemoKillsReferenceTheRightPlayers) {
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));
  CHECK(match.rounds.size() >= 2);
  if (match.rounds.size() < 2) return;
  CHECK(match.rounds[0].kills.size() >= 1);
  CHECK(match.rounds[1].kills.size() >= 2);
  if (match.rounds[0].kills.empty() || match.rounds[1].kills.size() < 2) return;

  const cs2mv::Kill& first = match.rounds[0].kills[0];
  CHECK(first.attacker >= 0);
  CHECK(first.victim >= 0);
  if (first.attacker < 0 || first.victim < 0) return;

  CHECK_EQ(match.players[first.attacker].name, std::string("Ada"));
  CHECK_EQ(match.players[first.victim].name, std::string("Cyd"));
  CHECK_EQ(first.weapon, std::string("m4a1"));
  CHECK(first.headshot);
  CHECK_EQ(first.attacker_team, cs2mv::kTeamCT);
  CHECK_EQ(first.victim_team, cs2mv::kTeamT);
  // No assister was involved, and the sentinel must not resolve to slot 0.
  CHECK_EQ(first.assister, -1);

  // The assisted kill in round two.
  const cs2mv::Kill& assisted = match.rounds[1].kills[1];
  CHECK(assisted.assister >= 0);
  if (assisted.assister >= 0) {
    CHECK_EQ(match.players[assisted.assister].name, std::string("Cyd"));
  }
}

TEST(DemoWarmupActivityIsDiscarded) {
  // The fixture stages a kill before begin_new_match. It must not appear.
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));
  int total_kills = 0;
  for (const cs2mv::Round& round : match.rounds) {
    total_kills += static_cast<int>(round.kills.size());
  }
  CHECK_EQ(total_kills, 7);   // 2 + 2 + 3, with the warmup kill dropped

  int scoreboard_kills = 0;
  for (const Player& p : match.players) scoreboard_kills += p.kills;
  CHECK_EQ(scoreboard_kills, 7);
}

TEST(DemoReaderUnpacksBzip2Demos) {
  Match packed;
  CHECK(ParseFixture("synthetic.dem.bz2", &packed));
  Match plain;
  CHECK(ParseFixture("synthetic.dem", &plain));

  CHECK_EQ(packed.map_name, plain.map_name);
  CHECK_EQ(packed.players.size(), plain.players.size());
  CHECK_EQ(packed.score_ct, plain.score_ct);
  CHECK_EQ(cs2mv::MatchToJson(packed), cs2mv::MatchToJson(plain));
}

TEST(DemoInventoryCountsMessages) {
  const std::string bytes = testing::ReadFileOrDie("synthetic.dem");
  DemoReader reader;
  std::string error;
  CHECK(reader.Init(bytes, &error));

  cs2mv::DemoInventory inventory;
  CHECK(cs2mv::InspectDemo(&reader, &inventory, &error));
  CHECK(inventory.frames[cs2mv::kDemPacket] > 0);
  CHECK(inventory.events["player_death"] == 8);  // 7 real plus the warmup one
  CHECK(inventory.events["round_end"] == 4);     // 3 real plus warmup
  CHECK(inventory.string_tables["userinfo"] == 1);
}

TEST(MatchJsonIsWellFormed) {
  Match match;
  CHECK(ParseFixture("synthetic.dem", &match));
  const std::string json = cs2mv::MatchToJson(match);

  CHECK(json.find("\"map\":\"de_dust2\"") != std::string::npos);
  CHECK(json.find("\"name\":\"Ada\"") != std::string::npos);
  CHECK(json.find("\"scoreCt\":2") != std::string::npos);

  // Balanced braces and brackets outside of strings.
  int braces = 0, brackets = 0;
  bool in_string = false, escaped = false;
  for (char c : json) {
    if (in_string) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') in_string = false;
      continue;
    }
    if (c == '"') in_string = true;
    else if (c == '{') ++braces;
    else if (c == '}') --braces;
    else if (c == '[') ++brackets;
    else if (c == ']') --brackets;
    CHECK(braces >= 0);
    CHECK(brackets >= 0);
  }
  CHECK_EQ(braces, 0);
  CHECK_EQ(brackets, 0);
  CHECK(!in_string);
}
