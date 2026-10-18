// A minimal PNG writer: enough to hand a browser an image without pulling in
// an image library. The pixel data goes into the zlib stream as stored
// (uncompressed) blocks, which every decoder accepts; the images this is used
// for are small and served over localhost, so size does not matter.
#ifndef CS2MV_PNG_H_
#define CS2MV_PNG_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cs2mv {

// Encodes an 8 bit greyscale image, `width * height` bytes, row major.
std::string EncodePngGray8(int width, int height, const std::vector<std::uint8_t>& pixels);

// Encodes an 8 bit RGBA image, `width * height * 4` bytes, row major.
std::string EncodePngRgba8(int width, int height, const std::vector<std::uint8_t>& pixels);

// CRC-32 as PNG uses it, exposed for tests.
std::uint32_t Crc32(const void* data, std::size_t size, std::uint32_t seed = 0);

}  // namespace cs2mv

#endif  // CS2MV_PNG_H_
