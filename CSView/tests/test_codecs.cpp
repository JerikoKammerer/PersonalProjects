#include <cstdio>
#include <string>

#include "cs2mv/bzip2.h"
#include "cs2mv/snappy.h"
#include "test_util.h"

using cs2mv::Bzip2Uncompress;
using cs2mv::IsBzip2;
using cs2mv::SnappyUncompress;

namespace {

std::string Bytes(const char* data, std::size_t size) {
  return std::string(data, size);
}

bool Uncompress(const std::string& input, std::string* out) {
  std::string error;
  const bool ok = SnappyUncompress(input.data(), input.size(), out, &error);
  if (!ok) std::fprintf(stderr, "    snappy: %s\n", error.c_str());
  return ok;
}

}  // namespace

// Vectors worked out by hand from the Snappy format description: one per tag
// type, including an overlapping copy and the extended literal length form.
TEST(SnappyDecodesGoldenVectors) {
  std::string out;

  CHECK(Uncompress(Bytes("\x03\x08" "abc", 5), &out));
  CHECK_EQ(out, std::string("abc"));

  // literal 'a', then copy1 with offset 1 and length 5: the copy reads bytes it
  // is still producing.
  CHECK(Uncompress(Bytes("\x06\x00\x61\x05\x01", 5), &out));
  CHECK_EQ(out, std::string("aaaaaa"));

  CHECK(Uncompress(Bytes("\x01\x00\x7A", 3), &out));
  CHECK_EQ(out, std::string("z"));

  // literal "abcd", then copy2 with offset 4 and length 6.
  CHECK(Uncompress(Bytes("\x0A\x0C" "abcd" "\x16\x04\x00", 9), &out));
  CHECK_EQ(out, std::string("abcdabcdab"));

  // 61 byte literal, which needs the one-extra-byte length form.
  std::string long_literal("\x3D\xF0\x3C", 3);
  long_literal.append(61, 'x');
  CHECK(Uncompress(long_literal, &out));
  CHECK_EQ(out, std::string(61, 'x'));
}

TEST(SnappyRejectsMalformedStreams) {
  std::string out;
  std::string error;
  // Length preamble promises four bytes, stream provides three.
  CHECK(!SnappyUncompress("\x04\x08" "abc", 5, &out, &error));
  CHECK(!error.empty());
  // Copy offset points before the start of the output.
  CHECK(!SnappyUncompress("\x05\x00\x61\x05\x09", 5, &out, &error));
  // Truncated literal.
  CHECK(!SnappyUncompress("\x0A\x24" "ab", 4, &out, &error));
}

TEST(Bzip2UnpacksMultiBlockStream) {
  const std::string compressed = testing::ReadFileOrDie("sample.bz2");
  CHECK(IsBzip2(compressed.data(), compressed.size()));

  std::string expected;
  for (int i = 0; i < 5000; ++i) {
    char line[80];
    std::snprintf(line, sizeof(line),
                  "line %d: the quick brown fox jumps over the lazy dog\n", i);
    expected += line;
  }

  std::string out;
  std::string error;
  CHECK(Bzip2Uncompress(compressed.data(), compressed.size(), &out, &error));
  CHECK_EQ(error, std::string());
  CHECK_EQ(out.size(), expected.size());
  CHECK(out == expected);
}

TEST(Bzip2RejectsNonBzipInput) {
  std::string out;
  std::string error;
  CHECK(!IsBzip2("hello", 5));
  CHECK(!Bzip2Uncompress("hello", 5, &out, &error));
  CHECK(!error.empty());

  // A valid header followed by rubbish must fail rather than loop or crash.
  const std::string broken = std::string("BZh9") + std::string(64, '\xA5');
  CHECK(!Bzip2Uncompress(broken.data(), broken.size(), &out, &error));
}
