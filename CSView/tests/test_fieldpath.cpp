// The field path Huffman tree decides how every entity update is read, and a
// single wrong weight or tie-break builds a different tree from the encoder's.
// It is worth checking on its own, because once it is wrong every later layer
// produces garbage with no clue as to why.
#include "cs2mv/fieldpath.h"

#include <set>
#include <string>

#include "cs2mv/bits.h"
#include "test_util.h"

using cs2mv::BitReader;
using cs2mv::FieldPathHuffman;

TEST(FieldPathTreeHasALeafPerOperation) {
  CHECK_EQ(FieldPathHuffman::Instance().LeafCount(),
           static_cast<int>(cs2mv::kFieldPathOpCount));
  CHECK_EQ(static_cast<int>(cs2mv::kFieldPathOpCount), 40);
}

TEST(FieldPathCodesArePrefixFreeAndUnique) {
  const FieldPathHuffman& huffman = FieldPathHuffman::Instance();
  std::set<std::string> codes;
  for (int op = 0; op < cs2mv::kFieldPathOpCount; ++op) {
    const std::string code = huffman.CodeFor(op);
    CHECK(!code.empty());
    CHECK(codes.insert(code).second);  // unique
  }
  // No code may be a prefix of another, or decoding could never be
  // unambiguous.
  for (const std::string& a : codes) {
    for (const std::string& b : codes) {
      if (a == b) continue;
      CHECK(a.compare(0, b.size(), b) != 0);
    }
  }
}

TEST(FieldPathCommonOperationsGetShortCodes) {
  const FieldPathHuffman& huffman = FieldPathHuffman::Instance();
  const std::size_t plus_one = huffman.CodeFor(cs2mv::kPlusOne).size();
  const std::size_t finish =
      huffman.CodeFor(cs2mv::kFieldPathEncodeFinish).size();
  const std::size_t rare = huffman.CodeFor(cs2mv::kPushN).size();

  // PlusOne is the heaviest operation by a wide margin, and "stop" the second.
  // Huffman guarantees they cannot be longer than a rare one.
  CHECK(plus_one <= finish);
  CHECK(finish < rare);
  // Sanity on magnitude: a 40 leaf tree cannot give the top symbol more than a
  // few bits, and the rarest ones sit far deeper.
  CHECK(plus_one <= 4);
  CHECK(rare >= 8);
}

TEST(FieldPathDecodesTheHeaviestOperationFromItsOwnCode) {
  // Feed the tree the exact bits of PlusOne's code and it must come back.
  const FieldPathHuffman& huffman = FieldPathHuffman::Instance();
  const std::string code = huffman.CodeFor(cs2mv::kPlusOne);

  unsigned char buffer[4] = {0, 0, 0, 0};
  for (std::size_t i = 0; i < code.size(); ++i) {
    if (code[i] == '1') buffer[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
  }
  BitReader bits(buffer, sizeof(buffer));
  CHECK_EQ(huffman.ReadOp(&bits), static_cast<int>(cs2mv::kPlusOne));
}

TEST(UBitVarFieldPathWidths) {
  // Hand-encoded against the cascade: one selector bit for 2 bits, then a
  // second for 4, and so on. Bits are least-significant first.
  {
    // selector 1, then 2 bits holding 3 -> 0b111
    const unsigned char data[] = {0x07};
    BitReader bits(data, sizeof(data));
    CHECK_EQ(bits.ReadUBitVarFieldPath(), 3u);
  }
  {
    // selector 0, selector 1, then 4 bits holding 5
    const unsigned char data[] = {0x16};
    BitReader bits(data, sizeof(data));
    CHECK_EQ(bits.ReadUBitVarFieldPath(), 5u);
  }
  {
    // selector 0, 0, selector 1, then 10 bits holding 1000
    // bits: b0=0 b1=0 b2=1, then value at bit 3
    unsigned char data[4] = {0, 0, 0, 0};
    data[0] |= 0x04;  // b2 = 1
    const unsigned value = 1000;
    for (int i = 0; i < 10; ++i) {
      if (value & (1u << i)) {
        const int bit = 3 + i;
        data[bit / 8] |= static_cast<unsigned char>(1u << (bit % 8));
      }
    }
    BitReader bits(data, sizeof(data));
    CHECK_EQ(bits.ReadUBitVarFieldPath(), 1000u);
  }
}

TEST(ZigzagVarIntRoundTrip) {
  // Zigzag maps -1,1,-2,2 onto 1,2,3,4; the encoder writes those as varints.
  const unsigned char data[] = {0x01, 0x02, 0x03, 0x04};
  BitReader bits(data, sizeof(data));
  CHECK_EQ(bits.ReadVarInt32(), -1);
  CHECK_EQ(bits.ReadVarInt32(), 1);
  CHECK_EQ(bits.ReadVarInt32(), -2);
  CHECK_EQ(bits.ReadVarInt32(), 2);
}
