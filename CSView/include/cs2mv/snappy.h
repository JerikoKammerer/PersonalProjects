// Snappy raw-block decompressor.
//
// CS2 compresses individual demo commands with Snappy (the DEM_IsCompressed
// bit in the command kind). This is the raw block format - a varint holding
// the uncompressed length followed by literal/copy tags - not the framed
// stream format.
#ifndef CS2MV_SNAPPY_H_
#define CS2MV_SNAPPY_H_

#include <cstddef>
#include <string>

namespace cs2mv {

// Decompresses `size` bytes at `data` into `*out` (replacing its contents).
// Returns false and fills `*error` (if non-null) on a malformed stream.
bool SnappyUncompress(const void* data, std::size_t size, std::string* out,
                      std::string* error);

// Reads just the length preamble, e.g. to size a buffer up front.
bool SnappyUncompressedLength(const void* data, std::size_t size,
                              std::size_t* length);

}  // namespace cs2mv

#endif  // CS2MV_SNAPPY_H_
