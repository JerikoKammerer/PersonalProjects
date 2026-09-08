// The match model the web UI renders.
#ifndef CS2MV_MATCH_H_
#define CS2MV_MATCH_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cs2mv {

enum Team {
  kTeamUnknown = 0,
  kTeamSpectator = 1,
  kTeamT = 2,
  kTeamCT = 3,
};

const char* TeamName(int team);

struct Player {
  int slot = -1;                 // index in the userinfo string table
  int user_id = -1;              // CMsgPlayerInfo.userid
  std::uint64_t steam_id = 0;    // 64 bit community id
  std::string name;
  int team = kTeamUnknown;       // team at the end of the match
  bool bot = false;
  bool hltv = false;

  int kills = 0;
  int deaths = 0;
  int assists = 0;
  int headshots = 0;
  int mvps = 0;
  int damage = 0;             // health damage dealt to enemies
  int utility_damage = 0;     // damage dealt with grenades
  int enemies_flashed = 0;
  int entry_kills = 0;        // first kill of a round
  int entry_deaths = 0;
  int rounds_played = 0;
  int score = 0;              // scoreboard score, when the demo reports it

  double kd() const { return deaths > 0 ? static_cast<double>(kills) / deaths : kills; }
  double adr() const {
    return rounds_played > 0 ? static_cast<double>(damage) / rounds_played : 0.0;
  }
  double hs_percent() const {
    return kills > 0 ? 100.0 * headshots / kills : 0.0;
  }
};

struct Kill {
  int tick = 0;
  double time = 0.0;  // seconds since the round started
  // Indices into Match::players; -1 when there was nobody (fall damage, bomb)
  // or when the player could not be resolved.
  int attacker = -1;
  int victim = -1;
  int assister = -1;
  // Sides at the moment of the kill, so a kill feed stays correct across the
  // halftime side swap.
  int attacker_team = kTeamUnknown;
  int victim_team = kTeamUnknown;
  std::string weapon;
  bool headshot = false;
  bool noscope = false;
  bool through_smoke = false;
  bool attacker_blind = false;
  bool wallbang = false;
  bool assist_flash = false;
};

// Round end reasons that matter for the UI; the numeric value is the one the
// round_end game event carries.
enum RoundEndReason {
  kReasonTargetBombed = 1,
  kReasonBombDefused = 7,
  kReasonCTWin = 8,
  kReasonTWin = 9,
  kReasonTargetSaved = 11,
};

const char* RoundEndReasonName(int reason);

struct Round {
  int number = 0;             // 1 based
  int start_tick = 0;
  int end_tick = 0;
  int winner = kTeamUnknown;
  int reason = 0;
  std::string reason_text;
  int score_t = 0;            // running score after this round
  int score_ct = 0;
  bool bomb_planted = false;
  bool bomb_defused = false;
  bool bomb_exploded = false;
  std::vector<Kill> kills;
};

struct Match {
  // Where it came from.
  std::string share_code;
  std::uint64_t match_id = 0;
  std::uint64_t outcome_id = 0;
  std::uint32_t token = 0;
  std::string demo_source;    // file path or URL the demo was read from

  // Demo header.
  std::string map_name;
  std::string server_name;
  std::string client_name;
  std::string demo_version;
  int build_number = 0;
  double tick_interval = 0.015625;  // 64 tick by default
  int playback_ticks = 0;
  double playback_time = 0.0;

  // Result.
  int score_t = 0;
  int score_ct = 0;
  int rounds_played = 0;

  std::vector<Player> players;
  std::vector<Round> rounds;

  // Anything the parser wanted to flag: unknown events, recoverable damage.
  std::vector<std::string> warnings;

  const Player* PlayerBySlot(int slot) const;
};

// Serialises the match for the web UI. `pretty` adds newlines and indentation.
std::string MatchToJson(const Match& m, bool pretty = false);

}  // namespace cs2mv

#endif  // CS2MV_MATCH_H_
