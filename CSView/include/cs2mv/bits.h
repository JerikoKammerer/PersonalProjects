// Little-endian bit reader for Source 2 network streams.
//
// Source 2 packs its message stream least-significant-bit first: the first bit
// read from a byte is that byte's bit 0, and a multi-bit field's first bit is
// the field's least significant bit.
#ifndef CS2MV_BITS_H_
#define CS2MV_BITS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace cs2mv {

class BitReader {
 public:
  BitReader(const void* data, std::size_t size)
      : data_(static_cast<const std::uint8_t*>(data)), size_(size) {}

  bool ok() const { return ok_; }

  // Bits not yet consumed. Bytes already pulled into the staging buffer count
  // as unconsumed, so this is exact.
  std::size_t BitsLeft() const { return (size_ - pos_) * 8 + avail_; }

  bool ReadBit() { return ReadBits(1) != 0; }

  // `count` must be in [0, 32].
  std::uint32_t ReadBits(int count) {
    if (count <= 0) return 0;
    if (count > 32) {
      Fail();
      return 0;
    }
    while (avail_ < count) {
      if (pos_ >= size_) {
        Fail();
        return 0;
      }
      buf_ |= static_cast<std::uint64_t>(data_[pos_++]) << avail_;
      avail_ += 8;
    }
    const std::uint32_t mask =
        (count == 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
    const std::uint32_t v = static_cast<std::uint32_t>(buf_) & mask;
    buf_ >>= count;
    avail_ -= count;
    return v;
  }

  // Source 2's "ubitvar": a 6 bit prefix whose top two bits select how many
  // extra bits carry the high part of the value. Used for message kinds.
  std::uint32_t ReadUBitVar() {
    std::uint32_t v = ReadBits(6);
    switch (v & 0x30) {
      case 0x10:
        v = (v & 0x0F) | (ReadBits(4) << 4);
        break;
      case 0x20:
        v = (v & 0x0F) | (ReadBits(8) << 4);
        break;
      case 0x30:
        v = (v & 0x0F) | (ReadBits(28) << 4);
        break;
      default:
        break;
    }
    return v;
  }

  // Protobuf-style varint, read a byte at a time out of the bit stream.
  std::uint32_t ReadVarUInt32() {
    std::uint32_t result = 0;
    for (int shift = 0; shift < 35; shift += 7) {
      const std::uint32_t b = ReadBits(8);
      if (!ok_) return 0;
      result |= (b & 0x7F) << shift;
      if ((b & 0x80) == 0) return result;
    }
    Fail();
    return 0;
  }

  // Reads `count` whole bytes. Fast path when the reader is byte aligned.
  bool ReadBytes(void* dst, std::size_t count) {
    auto* out = static_cast<std::uint8_t*>(dst);
    if (avail_ == 0 && pos_ + count <= size_) {
      std::memcpy(out, data_ + pos_, count);
      pos_ += count;
      return true;
    }
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = static_cast<std::uint8_t>(ReadBits(8));
      if (!ok_) return false;
    }
    return true;
  }

  bool ReadString(std::size_t count, std::string* out) {
    out->resize(count);
    if (count == 0) return true;
    return ReadBytes(&(*out)[0], count);
  }

 private:
  void Fail() { ok_ = false; }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  std::uint64_t buf_ = 0;  // staging bits, LSB is the next bit out
  int avail_ = 0;
  bool ok_ = true;
};

}  // namespace cs2mv

#endif  // CS2MV_BITS_H_
