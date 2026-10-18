// CS2 game coordinator match lookup: the messages that turn a share code into
// a demo URL.
//
// The exchange is:
//
//   client -> GC   k_EMsgGCCStrike15_v2_MatchListRequestFullGameInfo (9147)
//                  { matchid, outcomeid, token }   <- straight from a share code
//   GC -> client   k_EMsgGCCStrike15_v2_MatchList (9139)
//                  { matches: [ { matchid, matchtime, roundstatsall: [...] } ] }
//
// and the demo URL is the `map` field of the last roundstatsall entry, which
// for a finished match holds something like
// http://replay123.valve.net/730/003693....dem.bz2 rather than a map name.
//
// This header builds the request and parses the reply. It does NOT speak to
// Steam: those messages travel inside an authenticated Steam client session
// (see locator.h and README.md for how to bridge that).
#ifndef CS2MV_GC_H_
#define CS2MV_GC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "cs2mv/sharecode.h"

namespace cs2mv {

// From ECsgoGCMsg in Valve's cstrike15_gcmessages.proto.
constexpr std::uint32_t kMsgMatchListRequestFullGameInfo = 9147;
constexpr std::uint32_t kMsgMatchList = 9139;

// The base of the SteamID64 range for individual accounts; account ids in GC
// messages are 32 bit and need this added.
constexpr std::uint64_t kSteamId64Base = 76561197960265728ull;

// CMsgGCCStrike15_v2_MatchmakingServerRoundStats
struct GcRoundStats {
  std::uint64_t reservation_id = 0;
  // Per-round this is the map name; on the final entry of a finished match it
  // is the demo download URL.
  std::string map;
  int round = 0;
  int round_result = 0;
  int match_result = 0;
  int match_duration = 0;
  std::uint32_t max_rounds = 0;
  std::vector<std::uint32_t> account_ids;  // from reservation.account_ids
  std::vector<int> kills;
  std::vector<int> assists;
  std::vector<int> deaths;
  std::vector<int> scores;
  std::vector<int> enemy_kills;
  std::vector<int> enemy_headshots;
  std::vector<int> mvps;
  std::vector<int> team_scores;
};

// CDataGCCStrike15_v2_MatchInfo
struct GcMatchInfo {
  std::uint64_t match_id = 0;
  std::uint32_t match_time = 0;  // unix time
  std::uint32_t tv_port = 0;
  std::uint32_t server_ip = 0;
  std::vector<GcRoundStats> round_stats;

  // The demo URL, or empty when the reply did not carry one (matches expire
  // from Valve's replay servers after about 30 days).
  std::string DemoUrl() const;

  // Steam ids of the players, taken from the last round stats entry.
  std::vector<std::uint64_t> SteamIds() const;
};

// Serialises CMsgGCCStrike15_v2_MatchListRequestFullGameInfo
// { matchid = 1, outcomeid = 2, token = 3 }.
std::string BuildMatchListRequest(const ShareCode& code);

// Parses CMsgGCCStrike15_v2_MatchList { matches = 4 }.
bool ParseMatchList(const void* data, std::size_t size,
                    std::vector<GcMatchInfo>* out, std::string* error);

// Convenience for tests and for tools that fabricate a reply: serialises a
// MatchList containing a single match whose demo URL is `demo_url`.
std::string BuildMatchListReply(std::uint64_t match_id, std::uint32_t match_time,
                                const std::string& demo_url,
                                const std::vector<std::uint32_t>& account_ids);

}  // namespace cs2mv

#endif  // CS2MV_GC_H_
