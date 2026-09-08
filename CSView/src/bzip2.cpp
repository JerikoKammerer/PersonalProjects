#include "cs2mv/bzip2.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cs2mv {
namespace {

constexpr int kMaxGroups = 6;
constexpr int kGroupSize = 50;
constexpr int kMaxCodeLenBits = 23;
constexpr int kMaxSymbols = 258;  // 256 byte values + RUNA/RUNB + EOB

// Most significant bit first, which is the order bzip2 writes its fields in.
class BitIn {
 public:
  BitIn(const std::uint8_t* d, std::size_t n) : d_(d), n_(n) {}

  bool ok() const { return ok_; }
  std::size_t byte_pos() const { return pos_; }
  bool exhausted() const { return pos_ >= n_ && bit_ == 0; }

  std::uint32_t Bits(int count) {
    std::uint32_t v = 0;
    for (int i = 0; i < count; ++i) {
      if (pos_ >= n_) {
        ok_ = false;
        return 0;
      }
      const std::uint32_t b = (d_[pos_] >> (7 - bit_)) & 1u;
      v = (v << 1) | b;
      if (++bit_ == 8) {
        bit_ = 0;
        ++pos_;
      }
    }
    return v;
  }

  bool Bit() { return Bits(1) != 0; }

  // Discards the remainder of the current byte.
  void AlignToByte() {
    if (bit_ != 0) {
      bit_ = 0;
      ++pos_;
    }
  }

 private:
  const std::uint8_t* d_;
  std::size_t n_;
  std::size_t pos_ = 0;
  int bit_ = 0;
  bool ok_ = true;
};

// Canonical Huffman decoding tables, in the layout bzip2's own
// hbCreateDecodeTables produces.
struct HuffGroup {
  int min_len = 0;
  int max_len = 0;
  std::int32_t limit[kMaxCodeLenBits + 2] = {0};
  std::int32_t base[kMaxCodeLenBits + 2] = {0};
  std::int32_t perm[kMaxSymbols] = {0};
};

bool BuildGroup(const std::uint8_t* lengths, int symcount, HuffGroup* g) {
  int min_len = 32, max_len = 0;
  for (int s = 0; s < symcount; ++s) {
    min_len = std::min<int>(min_len, lengths[s]);
    max_len = std::max<int>(max_len, lengths[s]);
  }
  if (min_len < 1 || max_len > kMaxCodeLenBits) return false;
  g->min_len = min_len;
  g->max_len = max_len;

  // Symbols ordered by (code length, symbol), which is the order canonical
  // codes are assigned in.
  int pp = 0;
  for (int l = min_len; l <= max_len; ++l) {
    for (int s = 0; s < symcount; ++s) {
      if (lengths[s] == l) g->perm[pp++] = s;
    }
  }

  int count[kMaxCodeLenBits + 2] = {0};
  for (int s = 0; s < symcount; ++s) count[lengths[s]]++;

  std::int32_t cum = 0, code = 0;
  for (int l = min_len; l <= max_len; ++l) {
    const std::int32_t first = code;
    g->base[l] = first - cum;
    cum += count[l];
    code = (code + count[l]) << 1;
    g->limit[l] = first + count[l] - 1;
  }
  return true;
}

// Reads one symbol. Walks up code lengths until the accumulated value falls
// inside that length's assigned range.
int HuffDecode(BitIn* in, const HuffGroup& g, int symcount, bool* ok) {
  int l = g.min_len;
  std::int32_t v = static_cast<std::int32_t>(in->Bits(l));
  while (v > g.limit[l]) {
    if (++l > g.max_len) {
      *ok = false;
      return 0;
    }
    v = (v << 1) | static_cast<std::int32_t>(in->Bits(1));
    if (!in->ok()) {
      *ok = false;
      return 0;
    }
  }
  const std::int32_t idx = v - g.base[l];
  if (idx < 0 || idx >= symcount) {
    *ok = false;
    return 0;
  }
  return g.perm[idx];
}

bool Err(std::string* error, const char* msg) {
  if (error != nullptr) *error = msg;
  return false;
}

// Decodes one bzip2 stream starting at in->byte_pos(). Appends to *out.
bool DecodeStream(BitIn* in, std::string* out, std::string* error) {
  if (in->Bits(8) != 0x42 || in->Bits(8) != 0x5A || in->Bits(8) != 0x68) {
    return Err(error, "bzip2: bad signature");
  }
  const int level = static_cast<int>(in->Bits(8)) - 0x30;
  if (level < 1 || level > 9) return Err(error, "bzip2: bad block size level");
  const std::size_t max_block = static_cast<std::size_t>(level) * 100000u;

  std::vector<std::uint8_t> buf;   // BWT permutation input
  std::vector<std::uint32_t> tt;   // inverse BWT link table
  buf.reserve(max_block);

  for (;;) {
    const std::uint32_t hi = in->Bits(24);
    const std::uint32_t lo = in->Bits(24);
    if (!in->ok()) return Err(error, "bzip2: truncated stream");
    if (hi == 0x177245u && lo == 0x385090u) {
      in->Bits(32);  // stream CRC
      in->AlignToByte();
      return in->ok() ? true : Err(error, "bzip2: truncated stream footer");
    }
    if (hi != 0x314159u || lo != 0x265359u) {
      return Err(error, "bzip2: bad block magic");
    }
    in->Bits(32);  // block CRC, not verified
    if (in->Bit()) return Err(error, "bzip2: randomised blocks not supported");
    const std::uint32_t orig_ptr = in->Bits(24);

    // Symbol map: which byte values occur in this block.
    std::uint8_t used[256];
    int used_count = 0;
    const std::uint32_t map16 = in->Bits(16);
    for (int i = 0; i < 16; ++i) {
      if ((map16 & (0x8000u >> i)) == 0) continue;
      const std::uint32_t bits = in->Bits(16);
      for (int j = 0; j < 16; ++j) {
        if (bits & (0x8000u >> j)) used[used_count++] = static_cast<std::uint8_t>(i * 16 + j);
      }
    }
    if (!in->ok()) return Err(error, "bzip2: truncated symbol map");
    if (used_count == 0) return Err(error, "bzip2: empty symbol map");
    const int symcount = used_count + 2;

    // Selectors: which Huffman table each 50 symbol group uses, MTF coded.
    const int group_count = static_cast<int>(in->Bits(3));
    if (group_count < 2 || group_count > kMaxGroups) {
      return Err(error, "bzip2: bad group count");
    }
    const int nselectors = static_cast<int>(in->Bits(15));
    if (nselectors < 1) return Err(error, "bzip2: no selectors");
    std::vector<std::uint8_t> selectors(nselectors);
    {
      std::uint8_t mtf[kMaxGroups];
      for (int i = 0; i < group_count; ++i) mtf[i] = static_cast<std::uint8_t>(i);
      for (int i = 0; i < nselectors; ++i) {
        int j = 0;
        while (in->Bit()) {
          if (++j >= group_count || !in->ok()) return Err(error, "bzip2: bad selector");
        }
        const std::uint8_t v = mtf[j];
        for (int k = j; k > 0; --k) mtf[k] = mtf[k - 1];
        mtf[0] = v;
        selectors[i] = v;
      }
    }

    // Huffman code lengths, delta coded.
    std::vector<HuffGroup> groups(group_count);
    for (int gi = 0; gi < group_count; ++gi) {
      std::uint8_t lengths[kMaxSymbols];
      int l = static_cast<int>(in->Bits(5));
      for (int s = 0; s < symcount; ++s) {
        for (;;) {
          if (l < 1 || l > 20) return Err(error, "bzip2: bad code length");
          if (!in->Bit()) break;
          l += in->Bit() ? -1 : 1;
          if (!in->ok()) return Err(error, "bzip2: truncated code lengths");
        }
        lengths[s] = static_cast<std::uint8_t>(l);
      }
      if (!BuildGroup(lengths, symcount, &groups[gi])) {
        return Err(error, "bzip2: bad Huffman table");
      }
    }

    // Huffman + move-to-front + run-length decode into the BWT buffer.
    buf.clear();
    std::uint32_t byte_count[256] = {0};
    std::uint8_t mtf[256];
    for (int i = 0; i < used_count; ++i) mtf[i] = used[i];

    std::uint32_t run_pos = 0, run = 0;
    int group_idx = -1, group_pos = 0;
    const HuffGroup* g = nullptr;
    bool decode_ok = true;

    for (;;) {
      if (group_pos == 0) {
        if (++group_idx >= nselectors) return Err(error, "bzip2: ran out of selectors");
        group_pos = kGroupSize;
        g = &groups[selectors[group_idx]];
      }
      --group_pos;
      const int sym = HuffDecode(in, *g, symcount, &decode_ok);
      if (!decode_ok || !in->ok()) return Err(error, "bzip2: bad Huffman code");

      if (sym <= 1) {
        // RUNA/RUNB: a bijective base-2 run length, accumulated across symbols.
        if (run_pos == 0) {
          run_pos = 1;
          run = 0;
        }
        run += run_pos << sym;
        run_pos <<= 1;
        if (run > max_block) return Err(error, "bzip2: run length overflow");
        continue;
      }

      if (run_pos != 0) {
        run_pos = 0;
        if (buf.size() + run > max_block) return Err(error, "bzip2: block overflow");
        const std::uint8_t b = mtf[0];
        byte_count[b] += run;
        buf.insert(buf.end(), run, b);
      }

      if (sym == symcount - 1) break;  // end of block

      const int j = sym - 1;
      if (j >= used_count) return Err(error, "bzip2: MTF index out of range");
      const std::uint8_t b = mtf[j];
      for (int k = j; k > 0; --k) mtf[k] = mtf[k - 1];
      mtf[0] = b;

      if (buf.size() >= max_block) return Err(error, "bzip2: block overflow");
      byte_count[b]++;
      buf.push_back(b);
    }

    // Inverse Burrows-Wheeler transform.
    const std::size_t n = buf.size();
    if (n == 0) return Err(error, "bzip2: empty block");
    if (orig_ptr >= n) return Err(error, "bzip2: origPtr out of range");

    std::uint32_t cursor[256];
    std::uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
      cursor[i] = acc;
      acc += byte_count[i];
    }
    tt.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) tt[cursor[buf[i]]++] = static_cast<std::uint32_t>(i);

    // Final run-length pass: four equal bytes are followed by a repeat count.
    std::size_t pos = tt[orig_ptr];
    int prev = -1;
    int run_len = 0;
    out->reserve(out->size() + n);
    for (std::size_t i = 0; i < n; ++i) {
      const int cur = buf[pos];
      pos = tt[pos];
      if (run_len == 4) {
        out->append(static_cast<std::size_t>(cur), static_cast<char>(prev));
        run_len = 0;
        prev = -1;
        continue;
      }
      if (cur == prev) {
        ++run_len;
      } else {
        run_len = 1;
        prev = cur;
      }
      out->push_back(static_cast<char>(cur));
    }
  }
}

}  // namespace

bool IsBzip2(const void* data, std::size_t size) {
  if (size < 4) return false;
  const std::uint8_t* d = static_cast<const std::uint8_t*>(data);
  return d[0] == 'B' && d[1] == 'Z' && d[2] == 'h' && d[3] >= '1' && d[3] <= '9';
}

bool Bzip2Uncompress(const void* data, std::size_t size, std::string* out,
                     std::string* error) {
  const std::uint8_t* d = static_cast<const std::uint8_t*>(data);
  out->clear();
  std::size_t offset = 0;
  while (offset + 4 <= size && IsBzip2(d + offset, size - offset)) {
    BitIn in(d + offset, size - offset);
    if (!DecodeStream(&in, out, error)) return false;
    // Concatenated streams start on a byte boundary after the footer.
    const std::size_t consumed = in.byte_pos();
    if (consumed == 0) return Err(error, "bzip2: made no progress");
    offset += consumed;
  }
  if (offset == 0) return Err(error, "bzip2: not a bzip2 stream");
  return true;
}

}  // namespace cs2mv
