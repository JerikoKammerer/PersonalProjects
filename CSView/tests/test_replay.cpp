// The replay's output formats: the frame JSON the page plays back, and the
// PNG the map is drawn from. Both are built by hand here, since the real
// thing needs a retail demo.
#include "cs2mv/replay.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "cs2mv/png.h"
#include "test_util.h"

using cs2mv::Replay;
using cs2mv::ReplayFrame;
using cs2mv::PlayerSample;

namespace {

Replay SmallReplay() {
  Replay r;
  r.map = "de_test";
  r.tick_rate = 64;
  r.tick_step = 4;
  r.bounds_min[0] = -100.0f;
  r.bounds_min[1] = -100.0f;
  r.bounds_max[0] = 100.0f;
  r.bounds_max[1] = 100.0f;
  cs2mv::ReplayPlayer p;
  p.id = 3;
  p.name = "Ada";
  p.steam_id = 76561198000000001ull;
  r.players.push_back(p);
  r.weapons.push_back("ak47");
  for (int tick : {5000, 5004, 5008}) {
    ReplayFrame f;
    f.tick = tick;
    PlayerSample s;
    s.player = 0;
    s.x = 1234.5f;
    s.y = -8.5f;
    s.yaw = 90.0f;
    s.health = 100;
    s.team = 2;
    s.flags = cs2mv::kPlayerAlive;
    s.weapon = 0;
    f.players.push_back(s);
    r.frames.push_back(f);
  }
  return r;
}

std::uint32_t BigEndian(const std::string& s, std::size_t at) {
  return (static_cast<std::uint32_t>(static_cast<unsigned char>(s[at])) << 24) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(s[at + 1])) << 16) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(s[at + 2])) << 8) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(s[at + 3]));
}

}  // namespace

TEST(ReplayJsonKeepsCoordinatesWholeAndFiltersByTick) {
  const Replay r = SmallReplay();
  const std::string all = cs2mv::ReplayToJson(r, 0, 0);
  // Coordinates over 999 must not lose their fraction to a %g style format.
  CHECK(all.find("1234.5") != std::string::npos);
  CHECK(all.find("-8.5") != std::string::npos);
  CHECK(all.find("\"steamId\":\"76561198000000001\"") != std::string::npos);
  CHECK(all.find("\"map\":\"de_test\"") != std::string::npos);
  CHECK(all.find("[5000,[[") != std::string::npos);
  CHECK(all.find("[5008,[[") != std::string::npos);

  const std::string middle = cs2mv::ReplayToJson(r, 5002, 5006);
  CHECK(middle.find("[5004,[[") != std::string::npos);
  CHECK(middle.find("[5000,[[") == std::string::npos);
  CHECK(middle.find("[5008,[[") == std::string::npos);
}

TEST(Crc32MatchesTheKnownCheckValue) {
  // The standard check: CRC-32 of "123456789".
  CHECK_EQ(cs2mv::Crc32("123456789", 9), 0xCBF43926u);
}

TEST(PngEncoderWritesAWellFormedFile) {
  std::vector<std::uint8_t> pixels(3 * 2 * 4, 0);
  pixels[0] = 255;  // one red pixel, top left
  pixels[3] = 255;
  const std::string png = cs2mv::EncodePngRgba8(3, 2, pixels);

  CHECK(png.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0);
  // IHDR: length 13, then width and height big endian.
  CHECK_EQ(BigEndian(png, 8), 13u);
  CHECK(png.compare(12, 4, "IHDR") == 0);
  CHECK_EQ(BigEndian(png, 16), 3u);
  CHECK_EQ(BigEndian(png, 20), 2u);
  // The IHDR CRC, at 29, covers the type and the 13 bytes of data.
  CHECK_EQ(BigEndian(png, 29), cs2mv::Crc32(png.data() + 12, 17));
  // The file ends with an empty IEND chunk.
  CHECK(png.compare(png.size() - 8, 4, "IEND") == 0);
}

TEST(ReplayMapCoversTheGrid) {
  Replay r = SmallReplay();
  r.grid_cell = 16.0f;
  r.grid_w = 13;
  r.grid_h = 13;
  r.visits.assign(13 * 13, 0);
  r.visits[6 * 13 + 6] = 50;
  const std::string png = cs2mv::ReplayMapPng(r);
  CHECK(!png.empty());
  CHECK_EQ(BigEndian(png, 16), 13u);
  CHECK_EQ(BigEndian(png, 20), 13u);
}
