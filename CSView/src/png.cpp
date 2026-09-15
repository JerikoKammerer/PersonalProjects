#include "cs2mv/png.h"

#include <algorithm>
#include <cstring>

namespace cs2mv {
namespace {

std::uint32_t CrcTable(int n) {
  static std::uint32_t table[256];
  static bool ready = false;
  if (!ready) {
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    ready = true;
  }
  return table[n];
}

void PutU32(std::string* out, std::uint32_t v) {
  out->push_back(static_cast<char>((v >> 24) & 0xFF));
  out->push_back(static_cast<char>((v >> 16) & 0xFF));
  out->push_back(static_cast<char>((v >> 8) & 0xFF));
  out->push_back(static_cast<char>(v & 0xFF));
}

void PutChunk(std::string* out, const char* type, const std::string& body) {
  PutU32(out, static_cast<std::uint32_t>(body.size()));
  std::string typed = std::string(type, 4) + body;
  out->append(typed);
  PutU32(out, Crc32(typed.data(), typed.size()));
}

// A zlib stream made of stored deflate blocks, each at most 65535 bytes.
std::string ZlibStored(const std::string& raw) {
  std::string out;
  out.push_back(static_cast<char>(0x78));  // deflate, 32K window
  out.push_back(static_cast<char>(0x01));  // no preset dictionary, fastest
  std::size_t pos = 0;
  do {
    const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
    const bool last = pos + len >= raw.size();
    out.push_back(static_cast<char>(last ? 1 : 0));
    out.push_back(static_cast<char>(len & 0xFF));
    out.push_back(static_cast<char>((len >> 8) & 0xFF));
    out.push_back(static_cast<char>(~len & 0xFF));
    out.push_back(static_cast<char>((~len >> 8) & 0xFF));
    out.append(raw, pos, len);
    pos += len;
  } while (pos < raw.size());

  // Adler-32 of the uncompressed data, big endian.
  std::uint32_t a = 1, b = 0;
  for (unsigned char c : raw) {
    a = (a + c) % 65521;
    b = (b + a) % 65521;
  }
  PutU32(&out, (b << 16) | a);
  return out;
}

std::string Encode(int width, int height, int channels, int color_type,
                   const std::vector<std::uint8_t>& pixels) {
  std::string out("\x89PNG\r\n\x1a\n", 8);

  std::string ihdr;
  PutU32(&ihdr, static_cast<std::uint32_t>(width));
  PutU32(&ihdr, static_cast<std::uint32_t>(height));
  ihdr.push_back(8);                             // bit depth
  ihdr.push_back(static_cast<char>(color_type));  // 0 grey, 6 RGBA
  ihdr.push_back(0);                             // compression
  ihdr.push_back(0);                             // filter
  ihdr.push_back(0);                             // no interlace
  PutChunk(&out, "IHDR", ihdr);

  // Each scanline is prefixed with a filter byte; 0 means none.
  std::string raw;
  const std::size_t stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  raw.reserve((stride + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    raw.append(reinterpret_cast<const char*>(pixels.data()) + static_cast<std::size_t>(y) * stride,
               stride);
  }
  PutChunk(&out, "IDAT", ZlibStored(raw));
  PutChunk(&out, "IEND", std::string());
  return out;
}

}  // namespace

std::uint32_t Crc32(const void* data, std::size_t size, std::uint32_t seed) {
  std::uint32_t c = seed ^ 0xFFFFFFFFu;
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) c = CrcTable((c ^ p[i]) & 0xFF) ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::string EncodePngGray8(int width, int height, const std::vector<std::uint8_t>& pixels) {
  return Encode(width, height, 1, 0, pixels);
}

std::string EncodePngRgba8(int width, int height, const std::vector<std::uint8_t>& pixels) {
  return Encode(width, height, 4, 6, pixels);
}

}  // namespace cs2mv
