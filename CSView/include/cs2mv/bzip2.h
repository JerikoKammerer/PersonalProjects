// bzip2 decompressor.
//
// Valve's replay servers hand out demos as .dem.bz2, so downloading one means
// being able to unpack it. Only decompression is implemented, and only for
// non-randomised blocks (the randomisation feature was deprecated in bzip2
// 0.9.5 and no current compressor emits it).
#ifndef CS2MV_BZIP2_H_
#define CS2MV_BZIP2_H_

#include <cstddef>
#include <string>

namespace cs2mv {

// Decompresses a complete bzip2 stream into `*out`. Multi-stream files
// (several bzip2 streams concatenated, as `bzip2 -c a b > c` produces) are
// handled. Returns false and fills `*error` (if non-null) on bad input.
bool Bzip2Uncompress(const void* data, std::size_t size, std::string* out,
                     std::string* error);

// True if `data` starts with the bzip2 signature "BZh" plus a level digit.
bool IsBzip2(const void* data, std::size_t size);

}  // namespace cs2mv

#endif  // CS2MV_BZIP2_H_
