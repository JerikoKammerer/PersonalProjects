// Turns a demo into a Match.
//
// What this reads
// ---------------
// CS2 demos carry two independent descriptions of a match: a stream of
// "legacy" game events (kills, damage, round transitions - the same events a
// server plugin would subscribe to), and a full entity/field snapshot system
// that describes every networked property of every entity each tick.
//
// The parser here reads the first and ignores the second. Game events plus the
// userinfo string table are enough for a scoreboard, a round timeline and a
// kill feed, and they need no flattened-serializer decoding, which is by far
// the largest and most version-fragile part of a Source 2 demo. Positions,
// economy, weapons carried and anything else that only exists as entity state
// are therefore out of scope; see README.md.
#ifndef CS2MV_PARSER_H_
#define CS2MV_PARSER_H_

#include <map>
#include <string>

#include "cs2mv/demo.h"
#include "cs2mv/match.h"

namespace cs2mv {

struct ParseOptions {
  // Cap on Match::warnings, so a badly behaved demo cannot produce an
  // unbounded report.
  int max_warnings = 64;
  // Reset accumulated stats when the match formally starts, discarding warmup
  // and knife round activity. Turn off to see everything the demo contains.
  bool reset_on_match_start = true;
};

// Reads `reader` to the end and fills `*out`. Returns false only when the demo
// could not be read at all; recoverable problems land in Match::warnings.
bool ParseDemo(DemoReader* reader, const ParseOptions& options, Match* out,
               std::string* error);

// A census of what a demo contains. Useful when a demo does not parse as
// expected: it answers "which frames and messages are actually in here" and
// "which game events fired" without interpreting any of them.
struct DemoInventory {
  std::map<int, long long> frames;          // EDemoCommands  -> count
  std::map<int, long long> messages;        // net/svc/GE kind -> count
  std::map<std::string, long long> events;  // game event name -> count
  std::map<std::string, long long> string_tables;
  long long total_bytes = 0;
  long long compressed_frames = 0;
  int last_tick = 0;
};

bool InspectDemo(DemoReader* reader, DemoInventory* out, std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_PARSER_H_
