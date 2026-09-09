// A dependency-free protobuf wire-format reader and writer.
//
// The project deliberately does not link libprotobuf or generate code from
// Valve's .proto files: only a couple of dozen fields are ever touched, and
// reading them off the wire directly keeps the build to "a compiler, nothing
// else". Field numbers used by the rest of the code are documented at their
// call sites against the message they come from.
#ifndef CS2MV_PROTOBUF_H_
#define CS2MV_PROTOBUF_H_

#include <cstdint>
#include <string>

namespace cs2mv {
namespace pb {

enum WireType {
  kVarint = 0,
  kFixed64 = 1,
  kLengthDelimited = 2,
  kStartGroup = 3,
  kEndGroup = 4,
  kFixed32 = 5,
};

// A view over a byte range. Does not own its storage.
struct Slice {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;

  std::string ToString() const {
    return std::string(reinterpret_cast<const char*>(data), size);
  }
};

// Reads fields in stream order. Typical use:
//
//   pb::Reader r(data, size);
//   std::uint32_t field;
//   while (r.NextField(&field)) {
//     switch (field) {
//       case 1: name = r.ReadString(); break;
//       case 3: id = r.ReadVarint(); break;
//       default: break;            // NextField skips anything left behind
//     }
//   }
//   if (!r.ok()) { ... }
//
// A read that does not match the field's wire type, or that runs past the end
// of the buffer, marks the reader failed and returns a zero value; NextField
// then returns false. Callers may ignore intermediate errors and check ok()
// once at the end.
class Reader {
 public:
  Reader(const void* data, std::size_t size)
      : p_(static_cast<const std::uint8_t*>(data)), end_(p_ + size) {}
  explicit Reader(const Slice& s) : Reader(s.data, s.size) {}
  explicit Reader(const std::string& s) : Reader(s.data(), s.size()) {}

  bool ok() const { return ok_; }

  // Advances to the next field, skipping the current field's value if the
  // caller did not consume it. Returns false at end of buffer or on error.
  bool NextField(std::uint32_t* field);

  WireType wire_type() const { return wire_; }

  std::uint64_t ReadVarint();
  std::int64_t ReadInt64() { return static_cast<std::int64_t>(ReadVarint()); }
  std::int32_t ReadInt32() { return static_cast<std::int32_t>(ReadVarint()); }
  std::uint32_t ReadUInt32() { return static_cast<std::uint32_t>(ReadVarint()); }
  bool ReadBool() { return ReadVarint() != 0; }
  std::int32_t ReadSInt32();
  std::uint64_t ReadFixed64();
  std::uint32_t ReadFixed32();
  float ReadFloat();

  Slice ReadBytes();
  std::string ReadString() { return ReadBytes().ToString(); }

  // Number of bytes not yet consumed.
  std::size_t remaining() const { return static_cast<std::size_t>(end_ - p_); }

 private:
  void Fail() { ok_ = false; }
  bool SkipValue();

  const std::uint8_t* p_;
  const std::uint8_t* end_;
  WireType wire_ = kVarint;
  bool pending_ = false;  // current field's value not yet consumed
  bool ok_ = true;
};

// Appends protobuf-encoded fields to a std::string.
class Writer {
 public:
  void AddVarint(std::uint32_t field, std::uint64_t value);
  void AddBool(std::uint32_t field, bool value) { AddVarint(field, value ? 1 : 0); }
  void AddFixed64(std::uint32_t field, std::uint64_t value);
  void AddFixed32(std::uint32_t field, std::uint32_t value);
  void AddBytes(std::uint32_t field, const void* data, std::size_t size);
  void AddString(std::uint32_t field, const std::string& s) {
    AddBytes(field, s.data(), s.size());
  }
  void AddMessage(std::uint32_t field, const Writer& sub) {
    AddBytes(field, sub.data_.data(), sub.data_.size());
  }

  const std::string& data() const { return data_; }
  void Clear() { data_.clear(); }

 private:
  void PutVarint(std::uint64_t v);
  void PutTag(std::uint32_t field, WireType wt) {
    PutVarint((static_cast<std::uint64_t>(field) << 3) | wt);
  }

  std::string data_;
};

}  // namespace pb
}  // namespace cs2mv

#endif  // CS2MV_PROTOBUF_H_
