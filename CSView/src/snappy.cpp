#include "cs2mv/snappy.h"

#include <cstdint>
#include <cstring>

namespace cs2mv {
namespace {

bool ReadPreamble(const std::uint8_t* d, std::size_t size, std::size_t* len,
                  std::size_t* consumed) {
  std::uint64_t n = 0;
  std::size_t i = 0;
  for (int shift = 0; shift <= 28; shift += 7) {
    if (i >= size) return false;
    const std::uint8_t b = d[i++];
    n |= static_cast<std::uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) {
      *len = static_cast<std::size_t>(n);
      *consumed = i;
      return true;
    }
  }
  return false;  // more than five bytes: not a valid 32 bit length
}

bool Err(std::string* error, const char* msg) {
  if (error != nullptr) *error = msg;
  return false;
}

}  // namespace

bool SnappyUncompressedLength(const void* data, std::size_t size,
                              std::size_t* length) {
  std::size_t consumed = 0;
  return ReadPreamble(static_cast<const std::uint8_t*>(data), size, length,
                      &consumed);
}

bool SnappyUncompress(const void* data, std::size_t size, std::string* out,
                      std::string* error) {
  const std::uint8_t* d = static_cast<const std::uint8_t*>(data);
  std::size_t expected = 0;
  std::size_t i = 0;
  if (!ReadPreamble(d, size, &expected, &i)) {
    return Err(error, "snappy: bad length preamble");
  }

  out->clear();
  out->reserve(expected);

  while (i < size) {
    const std::uint8_t tag = d[i++];
    const int kind = tag & 0x03;

    if (kind == 0) {  // literal
      std::size_t len = tag >> 2;
      if (len >= 60) {
        const std::size_t extra = len - 59;
        if (i + extra > size) return Err(error, "snappy: truncated literal length");
        len = 0;
        for (std::size_t k = 0; k < extra; ++k) {
          len |= static_cast<std::size_t>(d[i + k]) << (8 * k);
        }
        i += extra;
      }
      ++len;
      if (i + len > size) return Err(error, "snappy: truncated literal");
      out->append(reinterpret_cast<const char*>(d + i), len);
      i += len;
      continue;
    }

    std::size_t len = 0;
    std::size_t offset = 0;
    if (kind == 1) {  // copy, 11 bit offset
      if (i + 1 > size) return Err(error, "snappy: truncated copy1");
      len = 4 + ((tag >> 2) & 0x07);
      offset = (static_cast<std::size_t>(tag >> 5) << 8) | d[i];
      i += 1;
    } else if (kind == 2) {  // copy, 16 bit offset
      if (i + 2 > size) return Err(error, "snappy: truncated copy2");
      len = (tag >> 2) + 1;
      offset = static_cast<std::size_t>(d[i]) |
               (static_cast<std::size_t>(d[i + 1]) << 8);
      i += 2;
    } else {  // copy, 32 bit offset
      if (i + 4 > size) return Err(error, "snappy: truncated copy4");
      len = (tag >> 2) + 1;
      offset = static_cast<std::size_t>(d[i]) |
               (static_cast<std::size_t>(d[i + 1]) << 8) |
               (static_cast<std::size_t>(d[i + 2]) << 16) |
               (static_cast<std::size_t>(d[i + 3]) << 24);
      i += 4;
    }

    if (offset == 0 || offset > out->size()) {
      return Err(error, "snappy: copy offset out of range");
    }
    // Copies may overlap their own output (that is how runs are encoded), so
    // this has to go one byte at a time rather than through memcpy.
    const std::size_t src = out->size() - offset;
    for (std::size_t k = 0; k < len; ++k) {
      const char c = (*out)[src + k];  // read before any possible reallocation
      out->push_back(c);
    }
  }

  if (out->size() != expected) {
    return Err(error, "snappy: decompressed size does not match preamble");
  }
  return true;
}

}  // namespace cs2mv
