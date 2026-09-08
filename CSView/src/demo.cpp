#include "cs2mv/demo.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "cs2mv/bzip2.h"
#include "cs2mv/protobuf.h"
#include "cs2mv/snappy.h"

namespace cs2mv {
namespace {

constexpr char kMagic[] = "PBDEMS2";
constexpr std::size_t kHeaderSize = 16;  // 8 byte magic + two int32

}  // namespace

const char* DemoCommandName(int kind) {
  switch (kind) {
    case kDemStop: return "DEM_Stop";
    case kDemFileHeader: return "DEM_FileHeader";
    case kDemFileInfo: return "DEM_FileInfo";
    case kDemSyncTick: return "DEM_SyncTick";
    case kDemSendTables: return "DEM_SendTables";
    case kDemClassInfo: return "DEM_ClassInfo";
    case kDemStringTables: return "DEM_StringTables";
    case kDemPacket: return "DEM_Packet";
    case kDemSignonPacket: return "DEM_SignonPacket";
    case kDemConsoleCmd: return "DEM_ConsoleCmd";
    case kDemCustomData: return "DEM_CustomData";
    case kDemCustomDataCallbacks: return "DEM_CustomDataCallbacks";
    case kDemUserCmd: return "DEM_UserCmd";
    case kDemFullPacket: return "DEM_FullPacket";
    case kDemSaveGame: return "DEM_SaveGame";
    case kDemSpawnGroups: return "DEM_SpawnGroups";
    case kDemAnimationData: return "DEM_AnimationData";
    case kDemAnimationHeader: return "DEM_AnimationHeader";
    case kDemRecovery: return "DEM_Recovery";
    default: return "DEM_Unknown";
  }
}

bool ReadDemoSummary(const std::string& path, DemoSummary* out,
                     std::string* error) {
  // The header frame sits at the very front and is a few hundred bytes; this
  // is generous enough to cover it without reading the whole demo.
  constexpr std::size_t kPrefixBytes = 128 * 1024;

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error != nullptr) *error = "cannot open " + path;
    return false;
  }
  std::string prefix(kPrefixBytes, '\0');
  f.read(&prefix[0], static_cast<std::streamsize>(kPrefixBytes));
  prefix.resize(static_cast<std::size_t>(f.gcount()));

  DemoReader reader;
  if (!reader.Init(std::move(prefix), error)) return false;

  DemoFrame frame;
  while (reader.Next(&frame)) {
    if (frame.kind != kDemFileHeader) continue;
    // CDemoFileHeader { server_name = 3, client_name = 4, map_name = 5,
    //                   build_num = 13 }
    pb::Reader r(frame.body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 3: out->server_name = r.ReadString(); break;
        case 4: out->client_name = r.ReadString(); break;
        case 5: out->map_name = r.ReadString(); break;
        case 13: out->build_number = r.ReadInt32(); break;
        default: break;
      }
    }
    return true;
  }
  if (error != nullptr) *error = "no header frame in " + path;
  return false;
}

bool DemoReader::Open(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error != nullptr) *error = "cannot open " + path;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  if (!f && !f.eof()) {
    if (error != nullptr) *error = "read error on " + path;
    return false;
  }
  return Init(ss.str(), error);
}

bool DemoReader::Init(std::string data, std::string* error) {
  error_.clear();
  pos_ = 0;
  stopped_ = false;

  if (IsBzip2(data.data(), data.size())) {
    std::string plain;
    std::string bz_err;
    if (!Bzip2Uncompress(data.data(), data.size(), &plain, &bz_err)) {
      error_ = bz_err;
      if (error != nullptr) *error = error_;
      return false;
    }
    data.swap(plain);
  }

  if (data.size() < kHeaderSize || data.compare(0, 7, kMagic) != 0) {
    error_ =
        "not a CS2 demo: expected a PBDEMS2 header. CS:GO era demos (HL2DEMO) "
        "use a different container and are not supported.";
    if (error != nullptr) *error = error_;
    return false;
  }

  data_.swap(data);
  pos_ = kHeaderSize;
  return true;
}

bool DemoReader::ReadVarint(std::uint32_t* out) {
  std::uint32_t result = 0;
  for (int shift = 0; shift < 35; shift += 7) {
    if (pos_ >= data_.size()) {
      error_ = "truncated demo: varint runs past end of file";
      return false;
    }
    const std::uint8_t b = static_cast<std::uint8_t>(data_[pos_++]);
    result |= static_cast<std::uint32_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) {
      *out = result;
      return true;
    }
  }
  error_ = "malformed demo: varint longer than five bytes";
  return false;
}

bool DemoReader::Next(DemoFrame* frame) {
  if (stopped_ || failed()) return false;
  if (pos_ >= data_.size()) return false;  // clean end of file

  std::uint32_t raw_kind = 0, tick = 0, size = 0;
  if (!ReadVarint(&raw_kind) || !ReadVarint(&tick) || !ReadVarint(&size)) {
    return false;
  }

  frame->compressed = (raw_kind & kDemIsCompressed) != 0;
  frame->kind = static_cast<int>(raw_kind & ~static_cast<std::uint32_t>(kDemIsCompressed));
  // A tick of 0xFFFFFFFF means "before the first tick"; keep it as -1.
  frame->tick = static_cast<std::int32_t>(tick);

  if (size > data_.size() - pos_) {
    error_ = "truncated demo: frame body runs past end of file";
    return false;
  }
  const char* body = data_.data() + pos_;
  pos_ += size;

  if (frame->compressed) {
    std::string snappy_err;
    if (!SnappyUncompress(body, size, &frame->body, &snappy_err)) {
      error_ = "frame " + std::string(DemoCommandName(frame->kind)) + ": " + snappy_err;
      return false;
    }
  } else {
    frame->body.assign(body, size);
  }

  if (frame->kind == kDemStop) {
    stopped_ = true;
    return false;
  }
  return true;
}

}  // namespace cs2mv
