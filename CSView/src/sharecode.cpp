#include "cs2mv/sharecode.h"

#include <cctype>
#include <cstring>

namespace cs2mv {
namespace {

constexpr int kAlphabetSize = 57;
constexpr int kPayloadChars = 25;
constexpr int kByteCount = 18;  // 144 bits

int AlphabetIndex(char c) {
  const char* p = std::strchr(kShareCodeAlphabet, c);
  if (p == nullptr || c == '\0') return -1;
  return static_cast<int>(p - kShareCodeAlphabet);
}

// Strips the optional "CSGO" prefix, all '-' separators and surrounding
// whitespace, leaving just the base-57 payload.
bool ExtractPayload(const std::string& code, std::string* payload,
                    std::string* error) {
  std::string s;
  s.reserve(code.size());
  for (char c : code) {
    if (std::isspace(static_cast<unsigned char>(c)) || c == '-') continue;
    s.push_back(c);
  }
  // Only strip the prefix when doing so leaves exactly one payload: every
  // character of "CSGO" is also a valid base-57 digit, so a bare payload can
  // legitimately start with those letters.
  if (s.size() == kPayloadChars + 4 && s.rfind("CSGO", 0) == 0) s.erase(0, 4);

  if (s.size() != kPayloadChars) {
    if (error != nullptr) {
      *error = "share code must have " + std::to_string(kPayloadChars) +
               " payload characters (CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx), got " +
               std::to_string(s.size());
    }
    return false;
  }
  for (char c : s) {
    if (AlphabetIndex(c) < 0) {
      if (error != nullptr) {
        *error = std::string("invalid character '") + c + "' in share code";
      }
      return false;
    }
  }
  *payload = s;
  return true;
}

std::uint64_t ReadLE(const unsigned char* p, int n) {
  std::uint64_t v = 0;
  for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

}  // namespace

const char kShareCodeAlphabet[] =
    "ABCDEFGHJKLMNOPQRSTUVWXYZabcdefhijkmnopqrstuvwxyz23456789";

bool DecodeShareCode(const std::string& code, ShareCode* out,
                     std::string* error) {
  std::string payload;
  if (!ExtractPayload(code, &payload, error)) return false;

  // Accumulate the base-57 digits into a 144-bit big-endian integer. The
  // encoding is little-digit-first, so the string is consumed in reverse.
  unsigned char buf[kByteCount] = {0};
  for (int i = kPayloadChars - 1; i >= 0; --i) {
    unsigned carry = static_cast<unsigned>(AlphabetIndex(payload[i]));
    for (int b = kByteCount - 1; b >= 0; --b) {
      unsigned v = buf[b] * kAlphabetSize + carry;
      buf[b] = static_cast<unsigned char>(v & 0xFF);
      carry = v >> 8;
    }
    if (carry != 0) {
      // Unreachable for 25 characters: 57^25 < 2^146 but the top two bits can
      // never be set by a code the game produced. Guard anyway.
      if (error != nullptr) *error = "share code overflows 144 bits";
      return false;
    }
  }

  // The 18 bytes hold three little-endian fields laid out back to back.
  out->match_id = ReadLE(buf + 0, 8);
  out->outcome_id = ReadLE(buf + 8, 8);
  out->token = static_cast<std::uint16_t>(ReadLE(buf + 16, 2));
  return true;
}

std::string EncodeShareCode(const ShareCode& sc) {
  unsigned char buf[kByteCount] = {0};
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<unsigned char>((sc.match_id >> (8 * i)) & 0xFF);
    buf[8 + i] = static_cast<unsigned char>((sc.outcome_id >> (8 * i)) & 0xFF);
  }
  buf[16] = static_cast<unsigned char>(sc.token & 0xFF);
  buf[17] = static_cast<unsigned char>((sc.token >> 8) & 0xFF);

  // Repeated division of the big-endian integer by 57 yields the digits
  // little-digit-first, which is the order they appear in the code.
  std::string digits;
  digits.reserve(kPayloadChars);
  for (int i = 0; i < kPayloadChars; ++i) {
    unsigned rem = 0;
    for (int b = 0; b < kByteCount; ++b) {
      unsigned v = (rem << 8) | buf[b];
      buf[b] = static_cast<unsigned char>(v / kAlphabetSize);
      rem = v % kAlphabetSize;
    }
    digits.push_back(kShareCodeAlphabet[rem]);
  }

  std::string out = "CSGO";
  for (int g = 0; g < 5; ++g) {
    out.push_back('-');
    out.append(digits, g * 5, 5);
  }
  return out;
}

bool NormalizeShareCode(const std::string& code, std::string* out,
                        std::string* error) {
  ShareCode sc;
  if (!DecodeShareCode(code, &sc, error)) return false;
  *out = EncodeShareCode(sc);
  return true;
}

bool ExtractShareCode(const std::string& text, std::string* code) {
  for (std::size_t start = text.find("CSGO-"); start != std::string::npos;
       start = text.find("CSGO-", start + 1)) {
    // Collect alphabet characters and separators until 25 payload characters
    // have been seen, then check that what was gathered actually decodes.
    std::string candidate;
    int payload = 0;
    for (std::size_t i = start; i < text.size() && payload < kPayloadChars; ++i) {
      const char c = text[i];
      if (c == '-') {
        candidate.push_back(c);
        continue;
      }
      if (AlphabetIndex(c) < 0) break;
      candidate.push_back(c);
      // The four letters of the "CSGO" prefix are themselves alphabet
      // characters, so they must not be counted towards the payload.
      if (candidate.size() > 5) ++payload;
    }
    if (payload < kPayloadChars) continue;

    ShareCode ignored;
    if (DecodeShareCode(candidate, &ignored, nullptr)) {
      *code = candidate;
      return true;
    }
  }

  // No prefix: maybe the text is (or contains) a bare code already.
  ShareCode ignored;
  if (DecodeShareCode(text, &ignored, nullptr)) {
    *code = text;
    return true;
  }
  return false;
}

}  // namespace cs2mv
