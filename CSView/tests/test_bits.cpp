// The bit reader underneath every decoder. Its position accounting has to be
// exact, because a one bit slip anywhere turns the rest of a packet to noise.
#include "cs2mv/bits.h"

#include <vector>

#include "test_util.h"

using cs2mv::BitReader;

TEST(BitReaderPositionMatchesBitsRead) {
  std::vector<unsigned char> data(64);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<unsigned char>(i * 37 + 11);
  }
  BitReader bits(data.data(), data.size());
  std::size_t consumed = 0;
  // A mix of widths, including the widest, in an awkward order.
  const int widths[] = {1, 32, 7, 32, 32, 32, 3, 32, 32, 32, 5, 1, 32, 16, 32};
  for (int width : widths) {
    bits.ReadBits(width);
    consumed += static_cast<std::size_t>(width);
    CHECK_EQ(bits.BitsConsumed(), consumed);
    CHECK_EQ(bits.BitsLeft(), data.size() * 8 - consumed);
  }
  CHECK(bits.ok());
}

TEST(BitReaderThirtyTwoBitReadsAreExact) {
  // Three 32 bit reads straddling byte boundaries must return the same bits
  // as reading them one at a time.
  std::vector<unsigned char> data(32);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<unsigned char>(i * 73 + 5);
  }
  BitReader wide(data.data(), data.size());
  BitReader narrow(data.data(), data.size());
  wide.ReadBits(5);
  narrow.ReadBits(5);
  for (int n = 0; n < 3; ++n) {
    const std::uint32_t w = wide.ReadBits(32);
    std::uint32_t v = 0;
    for (int i = 0; i < 32; ++i) v |= (narrow.ReadBit() ? 1u : 0u) << i;
    CHECK_EQ(w, v);
    CHECK_EQ(wide.BitsConsumed(), narrow.BitsConsumed());
  }
}
