#include "cs2mv/protobuf.h"

#include <cstring>

namespace cs2mv {
namespace pb {

bool Reader::NextField(std::uint32_t* field) {
  if (pending_ && !SkipValue()) return false;
  if (!ok_ || p_ >= end_) return false;

  const std::uint64_t tag = ReadVarint();
  if (!ok_) return false;
  const std::uint32_t wt = static_cast<std::uint32_t>(tag & 0x07);
  if (wt > kFixed32) {
    Fail();
    return false;
  }
  wire_ = static_cast<WireType>(wt);
  *field = static_cast<std::uint32_t>(tag >> 3);
  if (*field == 0) {
    Fail();
    return false;
  }
  pending_ = true;
  return true;
}

bool Reader::SkipValue() {
  pending_ = false;
  switch (wire_) {
    case kVarint:
      ReadVarint();
      break;
    case kFixed64:
      ReadFixed64();
      break;
    case kFixed32:
      ReadFixed32();
      break;
    case kLengthDelimited:
      ReadBytes();
      break;
    default:
      // Groups are not used by any message this project reads.
      Fail();
      break;
  }
  return ok_;
}

std::uint64_t Reader::ReadVarint() {
  // Called both for tags (where pending_ is not yet set) and for field values.
  pending_ = false;
  std::uint64_t result = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (p_ >= end_) {
      Fail();
      return 0;
    }
    const std::uint8_t b = *p_++;
    result |= static_cast<std::uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) return result;
  }
  Fail();
  return 0;
}

std::int32_t Reader::ReadSInt32() {
  const std::uint32_t v = static_cast<std::uint32_t>(ReadVarint());
  return static_cast<std::int32_t>((v >> 1) ^ (~(v & 1) + 1));
}

std::uint64_t Reader::ReadFixed64() {
  pending_ = false;
  if (end_ - p_ < 8) {
    Fail();
    return 0;
  }
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p_[i]) << (8 * i);
  p_ += 8;
  return v;
}

std::uint32_t Reader::ReadFixed32() {
  pending_ = false;
  if (end_ - p_ < 4) {
    Fail();
    return 0;
  }
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p_[i]) << (8 * i);
  p_ += 4;
  return v;
}

float Reader::ReadFloat() {
  const std::uint32_t bits = ReadFixed32();
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

Slice Reader::ReadBytes() {
  const std::uint64_t len = ReadVarint();
  if (!ok_) return Slice();
  if (len > static_cast<std::uint64_t>(end_ - p_)) {
    Fail();
    return Slice();
  }
  Slice s;
  s.data = p_;
  s.size = static_cast<std::size_t>(len);
  p_ += len;
  return s;
}

void Writer::PutVarint(std::uint64_t v) {
  while (v >= 0x80) {
    data_.push_back(static_cast<char>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  data_.push_back(static_cast<char>(v));
}

void Writer::AddVarint(std::uint32_t field, std::uint64_t value) {
  PutTag(field, kVarint);
  PutVarint(value);
}

void Writer::AddFixed64(std::uint32_t field, std::uint64_t value) {
  PutTag(field, kFixed64);
  for (int i = 0; i < 8; ++i) {
    data_.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  }
}

void Writer::AddFixed32(std::uint32_t field, std::uint32_t value) {
  PutTag(field, kFixed32);
  for (int i = 0; i < 4; ++i) {
    data_.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  }
}

void Writer::AddBytes(std::uint32_t field, const void* data, std::size_t size) {
  PutTag(field, kLengthDelimited);
  PutVarint(size);
  data_.append(static_cast<const char*>(data), size);
}

}  // namespace pb
}  // namespace cs2mv
