// The 2D replay: where everyone was, a few times a second, for the whole
// match - plus the grenades in the air, the bomb, and the shape of the map.
//
// All of it comes out of the entity state the demo carries (entities.h). No
// map images are shipped: the walkable area is learnt from where players
// actually stood, and the bomb sites from the trigger volumes the map
// declares, so any map works, including ones that did not exist when this
// was written.
#ifndef CS2MV_REPLAY_H_
#define CS2MV_REPLAY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "cs2mv/demo.h"

namespace cs2mv {

struct ReplayOptions {
  // Sample every N ticks. At 64 tick, 4 gives 16 frames a second, which is
  // smooth enough to interpolate between and keeps a match to a few
  // megabytes.
  int tick_step = 4;
};

struct ReplayPlayer {
  int id = -1;                // the controller's entity index: stable for the match
  std::string name;
  std::uint64_t steam_id = 0;
};

// Bits in PlayerSample::flags.
enum PlayerFlags {
  kPlayerAlive = 1,
  kPlayerHasBomb = 2,
  kPlayerDefusing = 4,
  kPlayerScoped = 8,
  kPlayerWalking = 16,
  kPlayerInBombZone = 32,
  kPlayerBlind = 64,
};

struct PlayerSample {
  int player = -1;            // index into Replay::players
  float x = 0, y = 0, z = 0;
  float yaw = 0;              // degrees, Source convention: 0 = +x, counter-clockwise
  int health = 0;
  int armor = 0;
  int team = 0;               // 2 = T, 3 = CT
  unsigned flags = 0;
  int weapon = -1;            // index into Replay::weapons
};

enum GrenadeKind {
  kGrenadeSmoke = 0,
  kGrenadeHE = 1,
  kGrenadeFlash = 2,
  kGrenadeMolotov = 3,
  kGrenadeDecoy = 4,
  kGrenadeFire = 5,           // one burning spot of an inferno
};

struct GrenadeSample {
  int kind = 0;
  float x = 0, y = 0, z = 0;
  bool active = false;        // a smoke that has bloomed, a fire that burns
  int thrower = -1;           // index into Replay::players, when known
};

enum BombState {
  kBombNone = 0,              // not in play, or carried (see the carrier)
  kBombDropped = 1,
  kBombPlanted = 2,
  kBombDefused = 3,
  kBombExploded = 4,
};

struct BombSample {
  int state = kBombNone;
  float x = 0, y = 0, z = 0;
  int carrier = -1;           // index into Replay::players
  int site = -1;              // 0 = A, 1 = B once planted
  bool being_defused = false;
};

struct ReplayFrame {
  int tick = 0;
  std::vector<PlayerSample> players;
  std::vector<GrenadeSample> grenades;
  BombSample bomb;
};

struct Bombsite {
  std::string letter;         // "A", "B", or "" when it never came up
  float min[3] = {0, 0, 0};
  float max[3] = {0, 0, 0};
};

struct Replay {
  std::string map;
  int tick_rate = 64;
  int tick_step = 4;
  // The radar's world extent, as the game rules entity declares it.
  float bounds_min[2] = {0, 0};
  float bounds_max[2] = {0, 0};
  std::vector<Bombsite> sites;
  std::vector<ReplayPlayer> players;
  std::vector<std::string> weapons;
  std::vector<ReplayFrame> frames;

  // Where players stood, on a grid of `grid_cell` units over the bounds:
  // the raw material for a map background.
  float grid_cell = 16.0f;
  int grid_w = 0, grid_h = 0;
  std::vector<std::uint16_t> visits;
};

// Runs the entity decoder over the whole demo. The reader must be positioned
// at the first frame (DemoReader::Rewind after any earlier pass).
bool BuildReplay(DemoReader* reader, const ReplayOptions& options, Replay* out,
                 std::string* error);

// Frames whose tick lies in [from_tick, to_tick] (all of them when both are
// zero), with the match-wide tables. Compact arrays, since a round is
// thousands of samples.
std::string ReplayToJson(const Replay& replay, int from_tick, int to_tick);

// A PNG of the map as the players revealed it: the walkable area in grey,
// shaded by how often it was walked, on a dark ground. Aligned to the
// replay's bounds, one pixel per grid cell.
std::string ReplayMapPng(const Replay& replay);

// The same grid with no blurring: a cell is opaque if anyone stood in it,
// transparent otherwise. The exact footprint, for lining a drawing of the
// map up with the world.
std::string ReplayWalkedPng(const Replay& replay);

}  // namespace cs2mv

#endif  // CS2MV_REPLAY_H_
