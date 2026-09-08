// Reader for the Source 2 demo container used by CS2 (.dem, "PBDEMS2").
//
// Layout:
//
//   char   magic[8]      "PBDEMS2\0"
//   int32  fileinfo_offset
//   int32  spawngroups_offset
//   frames...
//
// and each frame is
//
//   varint32 kind        EDemoCommands, OR'd with 64 when the body is Snappy
//                        compressed (DEM_IsCompressed)
//   varint32 tick
//   varint32 size
//   byte     body[size]  a serialised CDemo* protobuf message
//
// This reader hands back frames with the body already decompressed. What the
// bodies mean is parser.cpp's problem.
#ifndef CS2MV_DEMO_H_
#define CS2MV_DEMO_H_

#include <cstdint>
#include <string>

namespace cs2mv {

// EDemoCommands, from Valve's demo.proto.
enum DemoCommandKind {
  kDemError = -1,
  kDemStop = 0,
  kDemFileHeader = 1,
  kDemFileInfo = 2,
  kDemSyncTick = 3,
  kDemSendTables = 4,
  kDemClassInfo = 5,
  kDemStringTables = 6,
  kDemPacket = 7,
  kDemSignonPacket = 8,
  kDemConsoleCmd = 9,
  kDemCustomData = 10,
  kDemCustomDataCallbacks = 11,
  kDemUserCmd = 12,
  kDemFullPacket = 13,
  kDemSaveGame = 14,
  kDemSpawnGroups = 15,
  kDemAnimationData = 16,
  kDemAnimationHeader = 17,
  kDemRecovery = 18,
  kDemMax = 19,
  kDemIsCompressed = 64,
};

const char* DemoCommandName(int kind);

struct DemoFrame {
  int kind = kDemError;   // with the compression bit already stripped
  bool compressed = false;
  std::int32_t tick = 0;
  std::string body;       // decompressed protobuf payload
};

// The handful of facts in a demo's first frame.
struct DemoSummary {
  std::string map_name;
  std::string server_name;
  std::string client_name;
  int build_number = 0;
};

// Reads just the CDemoFileHeader, which is the first frame in the file, so
// this costs a few kilobytes rather than the few hundred megabytes a full
// Init would. Used to label a list of demos without parsing any of them.
bool ReadDemoSummary(const std::string& path, DemoSummary* out,
                     std::string* error);

// Iterates the frames of a demo held in memory.
class DemoReader {
 public:
  DemoReader() = default;

  // Takes ownership of `data`, which must be a complete .dem file. If it looks
  // like a bzip2 stream it is unpacked first, so a downloaded .dem.bz2 can be
  // handed over as is.
  bool Init(std::string data, std::string* error);

  // Reads the whole file at `path` and calls Init.
  bool Open(const std::string& path, std::string* error);

  // Fills `*frame` with the next frame. Returns false at DEM_Stop, at end of
  // file, or on error - check error() to tell the difference.
  bool Next(DemoFrame* frame);

  const std::string& error() const { return error_; }
  bool failed() const { return !error_.empty(); }

  // Byte offset of the current read position, for progress reporting.
  std::size_t position() const { return pos_; }
  std::size_t size() const { return data_.size(); }

 private:
  bool ReadVarint(std::uint32_t* out);

  std::string data_;
  std::size_t pos_ = 0;
  std::string error_;
  bool stopped_ = false;
};

}  // namespace cs2mv

#endif  // CS2MV_DEMO_H_
