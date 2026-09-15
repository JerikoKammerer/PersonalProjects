// CS2 / CS:GO match share code encoding and decoding.
//
// A share code is a base-57 encoding of a 144-bit value carrying three fields:
//
//   CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx   (25 payload characters, 5 groups of 5)
//
//   match_id    uint64  the GC's id for the match
//   outcome_id  uint64  aka reservation id, identifies the match outcome record
//   token       uint16  aka tv_port, needed to authorise the lookup
//
// The three fields are what the CS2 game coordinator wants in a
// CMsgGCCStrike15_v2_MatchListRequestFullGameInfo request; see gc.h.
#ifndef CS2MV_SHARECODE_H_
#define CS2MV_SHARECODE_H_

#include <cstdint>
#include <string>

namespace cs2mv {

// The 57 character alphabet Valve uses. Visually ambiguous characters
// (I, l, g, 0, 1) are excluded.
extern const char kShareCodeAlphabet[];

struct ShareCode {
  std::uint64_t match_id = 0;
  std::uint64_t outcome_id = 0;
  std::uint16_t token = 0;
};

// Decodes `code` into `*out`. Accepts codes with or without the "CSGO-" prefix
// and with or without group separators, e.g. all of these are the same code:
//
//   CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA
//   Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA
//   Cji4ZrQMJJs6JyqovwoSmJkDA
//
// Leading and trailing whitespace is ignored. Returns false and fills `*error`
// (if non-null) when the code is malformed.
bool DecodeShareCode(const std::string& code, ShareCode* out, std::string* error);

// Inverse of DecodeShareCode. Always produces the canonical, dashed form.
std::string EncodeShareCode(const ShareCode& sc);

// Rewrites `code` into its canonical form (CSGO- prefix, five dashed groups).
// Useful for logging and for using a code as a cache key.
bool NormalizeShareCode(const std::string& code, std::string* out,
                        std::string* error);

// Pulls a share code out of surrounding text. CS2's "copy" button hands out a
// whole Steam URL rather than a bare code:
//
//   steam://rungame/730/76561202255233023/+csgo_download_match%20CSGO-xxxxx-...
//
// so accept that, a chat message, or anything else with a code embedded in it.
// Returns false when no well-formed code is present.
bool ExtractShareCode(const std::string& text, std::string* code);

}  // namespace cs2mv

#endif  // CS2MV_SHARECODE_H_
