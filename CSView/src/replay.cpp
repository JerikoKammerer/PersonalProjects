#include "cs2mv/replay.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>

#include "cs2mv/entities.h"
#include "cs2mv/json.h"
#include "cs2mv/png.h"
#include "cs2mv/protobuf.h"

namespace cs2mv {
namespace {

// The world is 32 cells of 512 units on each axis, centred on the origin.
constexpr float kCellSize = 512.0f;
constexpr float kWorldHalf = 16384.0f;

// An entity handle: 14 bits of index, then a serial number. 0xFFFFFF is
// "none".
int HandleIndex(const FieldValue* handle) {
  if (handle == nullptr) return -1;
  const unsigned long long h = handle->u;
  if (h >= 0xFFFFFFull) return -1;
  return static_cast<int>(h & 0x3FFF);
}

bool EntityPosition(const Entity& e, float* x, float* y, float* z) {
  const FieldValue* cx = e.Get("CBodyComponent.m_cellX");
  const FieldValue* cy = e.Get("CBodyComponent.m_cellY");
  const FieldValue* cz = e.Get("CBodyComponent.m_cellZ");
  if (cx == nullptr || cy == nullptr) return false;
  const FieldValue* vx = e.Get("CBodyComponent.m_vecX");
  const FieldValue* vy = e.Get("CBodyComponent.m_vecY");
  const FieldValue* vz = e.Get("CBodyComponent.m_vecZ");
  *x = static_cast<float>(cx->AsInt()) * kCellSize - kWorldHalf + (vx ? vx->AsFloat() : 0.0f);
  *y = static_cast<float>(cy->AsInt()) * kCellSize - kWorldHalf + (vy ? vy->AsFloat() : 0.0f);
  *z = cz ? static_cast<float>(cz->AsInt()) * kCellSize - kWorldHalf + (vz ? vz->AsFloat() : 0.0f)
          : 0.0f;
  return true;
}

int IntOf(const Entity& e, const char* name, int fallback = 0) {
  const FieldValue* v = e.Get(name);
  return v == nullptr ? fallback : static_cast<int>(v->AsInt());
}

bool BoolOf(const Entity& e, const char* name) { return IntOf(e, name) != 0; }

// "CWeaponAK47" -> "ak47", "CAK47" -> "ak47", "CKnife" -> "knife".
std::string WeaponName(const std::string& class_name) {
  std::string s = class_name;
  if (s.rfind("CWeapon", 0) == 0) {
    s = s.substr(7);
  } else if (!s.empty() && s[0] == 'C') {
    s = s.substr(1);
  }
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

int GrenadeKindOf(const std::string& class_name) {
  if (class_name == "CSmokeGrenadeProjectile") return kGrenadeSmoke;
  if (class_name == "CHEGrenadeProjectile") return kGrenadeHE;
  if (class_name == "CFlashbangProjectile") return kGrenadeFlash;
  if (class_name == "CMolotovProjectile") return kGrenadeMolotov;
  if (class_name == "CDecoyProjectile") return kGrenadeDecoy;
  return -1;
}

// Accumulates the replay while the decoder runs.
class Builder {
 public:
  Builder(const ReplayOptions& options, Replay* out) : options_(options), out_(out) {}

  void Sample(int tick, const EntityDecoder& decoder) {
    ReplayFrame frame;
    frame.tick = tick;

    // One walk over the entities sorts out the few that matter; the class
    // of each is looked up by id, so this costs a byte per entity rather
    // than a string comparison.
    std::vector<const Entity*> controllers, c4s, placed;
    for (const auto& entry : decoder.entities()) {
      const Entity& e = entry.second;
      switch (KindOf(e)) {
        case kKindController: controllers.push_back(&e); break;
        case kKindC4: c4s.push_back(&e); break;
        case kKindPlaced: placed.push_back(&e); break;
        default: break;
      }
    }

    // Pass one: who is who. Controllers name the players; their pawn handle
    // says which pawn entity is theirs right now.
    std::map<int, int> pawn_to_player;  // pawn entity index -> player index
    for (const Entity* ep : controllers) {
      const Entity& e = *ep;
      const int pawn = HandleIndex(e.Get("m_hPlayerPawn"));
      const FieldValue* name = e.Get("m_iszPlayerName");
      const FieldValue* steam = e.Get("m_steamID");
      const std::uint64_t steam_id = steam ? steam->u : 0;
      if (pawn < 0 && steam_id == 0) continue;
      // The recorder's own controller has neither a pawn nor a Steam id.
      auto it = player_index_.find(e.index);
      if (it == player_index_.end()) {
        if (pawn < 0) continue;
        ReplayPlayer p;
        p.id = e.index;
        it = player_index_.emplace(e.index, static_cast<int>(out_->players.size())).first;
        out_->players.push_back(p);
      }
      ReplayPlayer& p = out_->players[static_cast<std::size_t>(it->second)];
      if (name != nullptr && !name->s.empty()) p.name = name->s;
      if (steam_id != 0) p.steam_id = steam_id;
      if (pawn >= 0) pawn_to_player[pawn] = it->second;
    }

    // Pass two: the bomb carrier, from the C4 weapon's owner.
    int bomb_carrier_pawn = -1;
    for (const Entity* ep : c4s) {
      const Entity& e = *ep;
      bomb_carrier_pawn = HandleIndex(e.Get("m_hOwnerEntity"));
      if (bomb_carrier_pawn < 0) {
        // Lying on the ground.
        float x, y, z;
        if (EntityPosition(e, &x, &y, &z)) {
          frame.bomb.state = kBombDropped;
          frame.bomb.x = x;
          frame.bomb.y = y;
          frame.bomb.z = z;
        }
      }
    }

    // Pass three: everything with a place on the map.
    std::set<int> sampled_players;
    for (const Entity* ep : placed) {
      const Entity& e = *ep;
      if (e.class_name == "CCSPlayerPawn") {
        auto owner = pawn_to_player.find(e.index);
        if (owner == pawn_to_player.end()) continue;
        PlayerSample s;
        s.player = owner->second;
        if (!EntityPosition(e, &s.x, &s.y, &s.z)) continue;
        const FieldValue* angles = e.Get("m_angEyeAngles");
        s.yaw = angles ? angles->v[1] : 0.0f;
        s.health = IntOf(e, "m_iHealth");
        s.armor = IntOf(e, "m_ArmorValue");
        s.team = IntOf(e, "m_iTeamNum");
        const bool alive = IntOf(e, "m_lifeState") == 0 && s.health > 0;
        if (alive) s.flags |= kPlayerAlive;
        if (e.index == bomb_carrier_pawn) {
          s.flags |= kPlayerHasBomb;
          frame.bomb.carrier = s.player;
        }
        if (BoolOf(e, "m_bIsDefusing")) s.flags |= kPlayerDefusing;
        if (BoolOf(e, "m_bIsScoped")) s.flags |= kPlayerScoped;
        if (BoolOf(e, "m_bIsWalking")) s.flags |= kPlayerWalking;
        if (BoolOf(e, "m_bInBombZone")) s.flags |= kPlayerInBombZone;
        const int weapon_entity = HandleIndex(e.Get("m_pWeaponServices.m_hActiveWeapon"));
        if (weapon_entity >= 0) {
          auto w = decoder.entities().find(weapon_entity);
          if (w != decoder.entities().end()) s.weapon = WeaponIndex(w->second.class_name);
        }
        if (alive) {
          Visit(s.x, s.y, s.z);
          const int zone = IntOf(e, "m_nWhichBombZone");
          if (zone > 0) NoteBombZone(s.x, s.y, s.z, zone);
        }
        last_sample_[s.player] = s;
        sampled_players.insert(s.player);
        frame.players.push_back(s);
        continue;
      }

      const int grenade = GrenadeKindOf(e.class_name);
      if (grenade >= 0) {
        GrenadeSample g;
        g.kind = grenade;
        if (!EntityPosition(e, &g.x, &g.y, &g.z)) continue;
        if (grenade == kGrenadeSmoke) {
          g.active = BoolOf(e, "m_bDidSmokeEffect");
          if (g.active) {
            // Smoke blooms where it detonated, which is not where the
            // projectile entity ends up.
            const FieldValue* at = e.Get("m_vSmokeDetonationPos");
            if (at != nullptr && (at->v[0] != 0.0f || at->v[1] != 0.0f)) {
              g.x = at->v[0];
              g.y = at->v[1];
              g.z = at->v[2];
            }
          }
        }
        auto thrower = pawn_to_player.find(HandleIndex(e.Get("m_hThrower")));
        if (thrower != pawn_to_player.end()) g.thrower = thrower->second;
        frame.grenades.push_back(g);
        continue;
      }

      if (e.class_name == "CInferno") {
        const int count = std::min(64, IntOf(e, "m_fireCount"));
        for (int i = 0; i < count; ++i) {
          const std::string suffix = "." + std::to_string(i);
          const FieldValue* at = e.Get("m_firePositions" + suffix);
          if (at == nullptr) continue;
          GrenadeSample g;
          g.kind = kGrenadeFire;
          g.x = at->v[0];
          g.y = at->v[1];
          g.z = at->v[2];
          const FieldValue* burning = e.Get("m_bFireIsBurning" + suffix);
          g.active = burning == nullptr || burning->AsInt() != 0;
          frame.grenades.push_back(g);
        }
        continue;
      }

      if (e.class_name == "CPlantedC4") {
        float x, y, z;
        if (!EntityPosition(e, &x, &y, &z)) continue;
        frame.bomb.x = x;
        frame.bomb.y = y;
        frame.bomb.z = z;
        frame.bomb.carrier = -1;
        frame.bomb.site = IntOf(e, "m_nBombSite", -1);
        frame.bomb.being_defused = BoolOf(e, "m_bBeingDefused");
        if (BoolOf(e, "m_bHasExploded")) {
          frame.bomb.state = kBombExploded;
        } else if (BoolOf(e, "m_bBombDefused")) {
          frame.bomb.state = kBombDefused;
        } else {
          frame.bomb.state = kBombPlanted;
        }
        continue;
      }

      if (e.class_name == "CBombTarget" && !sites_seen_.count(e.index)) {
        // A trigger volume: an origin plus a box around it.
        float x, y, z;
        const FieldValue* mins = e.Get("m_vecMins");
        const FieldValue* maxs = e.Get("m_vecMaxs");
        if (EntityPosition(e, &x, &y, &z) && mins != nullptr && maxs != nullptr) {
          Bombsite site;
          site.min[0] = x + mins->v[0];
          site.min[1] = y + mins->v[1];
          site.min[2] = z + mins->v[2];
          site.max[0] = x + maxs->v[0];
          site.max[1] = y + maxs->v[1];
          site.max[2] = z + maxs->v[2];
          out_->sites.push_back(site);
          site_zone_votes_.emplace_back();
          sites_seen_.insert(e.index);
        }
        continue;
      }

      if (e.class_name == "CCSGameRulesProxy" && !have_bounds_) {
        const FieldValue* mins = e.Get("m_pGameRules.m_vMinimapMins");
        const FieldValue* maxs = e.Get("m_pGameRules.m_vMinimapMaxs");
        if (mins != nullptr && maxs != nullptr && maxs->v[0] > mins->v[0]) {
          out_->bounds_min[0] = mins->v[0];
          out_->bounds_min[1] = mins->v[1];
          out_->bounds_max[0] = maxs->v[0];
          out_->bounds_max[1] = maxs->v[1];
          have_bounds_ = true;
          InitGrid();
        }
      }
    }

    // Players whose pawn is gone (dead and cleaned up, or spectating) keep
    // their last known place, marked dead, so the map still shows where
    // they fell.
    for (const auto& known : last_sample_) {
      if (sampled_players.count(known.first)) continue;
      PlayerSample s = known.second;
      s.flags &= ~static_cast<unsigned>(kPlayerAlive | kPlayerHasBomb | kPlayerDefusing);
      s.health = 0;
      frame.players.push_back(s);
    }
    std::sort(frame.players.begin(), frame.players.end(),
              [](const PlayerSample& a, const PlayerSample& b) { return a.player < b.player; });

    out_->frames.push_back(std::move(frame));
  }

  void Finish() {
    // Bomb site letters: the game tells each pawn which zone it stands in
    // (1 = A, 2 = B), so whichever letter was reported most from inside a
    // site's box is its name.
    for (std::size_t i = 0; i < out_->sites.size(); ++i) {
      const auto& votes = site_zone_votes_[i];
      int best = 0, best_count = 0;
      for (const auto& v : votes) {
        if (v.second > best_count) {
          best = v.first;
          best_count = v.second;
        }
      }
      if (best == 1) out_->sites[i].letter = "A";
      if (best == 2) out_->sites[i].letter = "B";
    }
    if (!have_bounds_) {
      // No radar extent from the game rules: pad the visited area instead.
      out_->bounds_min[0] = seen_min_[0] - 256.0f;
      out_->bounds_min[1] = seen_min_[1] - 256.0f;
      out_->bounds_max[0] = seen_max_[0] + 256.0f;
      out_->bounds_max[1] = seen_max_[1] + 256.0f;
      InitGrid();
      for (const ReplayFrame& frame : out_->frames) {
        for (const PlayerSample& s : frame.players) {
          if (s.flags & kPlayerAlive) Visit(s.x, s.y, s.z);
        }
      }
    }
  }

 private:
  enum EntityKind { kKindOther = 0, kKindController, kKindC4, kKindPlaced };

  EntityKind KindOf(const Entity& e) {
    if (e.class_id < 0) return kKindOther;
    const std::size_t id = static_cast<std::size_t>(e.class_id);
    if (id >= kind_by_class_.size()) kind_by_class_.resize(id + 1, -1);
    if (kind_by_class_[id] < 0) {
      const std::string& n = e.class_name;
      EntityKind kind = kKindOther;
      if (n == "CCSPlayerController") {
        kind = kKindController;
      } else if (n == "CC4") {
        kind = kKindC4;
      } else if (n == "CCSPlayerPawn" || n == "CInferno" || n == "CPlantedC4" ||
                 n == "CBombTarget" || n == "CCSGameRulesProxy" || GrenadeKindOf(n) >= 0) {
        kind = kKindPlaced;
      }
      kind_by_class_[id] = static_cast<signed char>(kind);
    }
    return static_cast<EntityKind>(kind_by_class_[id]);
  }

  int WeaponIndex(const std::string& class_name) {
    auto it = weapon_index_.find(class_name);
    if (it != weapon_index_.end()) return it->second;
    const int index = static_cast<int>(out_->weapons.size());
    out_->weapons.push_back(WeaponName(class_name));
    weapon_index_[class_name] = index;
    return index;
  }

  void InitGrid() {
    out_->grid_w = static_cast<int>(
        std::ceil((out_->bounds_max[0] - out_->bounds_min[0]) / out_->grid_cell));
    out_->grid_h = static_cast<int>(
        std::ceil((out_->bounds_max[1] - out_->bounds_min[1]) / out_->grid_cell));
    if (out_->grid_w < 1) out_->grid_w = 1;
    if (out_->grid_h < 1) out_->grid_h = 1;
    out_->visits.assign(static_cast<std::size_t>(out_->grid_w) * out_->grid_h, 0);
    out_->visit_top.assign(out_->visits.size(), -1e30f);
  }

  void Visit(float x, float y, float z) {
    seen_min_[0] = std::min(seen_min_[0], x);
    seen_min_[1] = std::min(seen_min_[1], y);
    seen_max_[0] = std::max(seen_max_[0], x);
    seen_max_[1] = std::max(seen_max_[1], y);
    if (out_->visits.empty()) return;
    const int gx = static_cast<int>((x - out_->bounds_min[0]) / out_->grid_cell);
    const int gy = static_cast<int>((y - out_->bounds_min[1]) / out_->grid_cell);
    if (gx < 0 || gy < 0 || gx >= out_->grid_w || gy >= out_->grid_h) return;
    const std::size_t at = static_cast<std::size_t>(gy) * out_->grid_w + gx;
    if (out_->visits[at] < 0xFFFF) ++out_->visits[at];
    out_->visit_top[at] = std::max(out_->visit_top[at], z);
  }

  void NoteBombZone(float x, float y, float z, int zone) {
    for (std::size_t i = 0; i < out_->sites.size(); ++i) {
      const Bombsite& s = out_->sites[i];
      if (x >= s.min[0] && x <= s.max[0] && y >= s.min[1] && y <= s.max[1] &&
          z >= s.min[2] - 64.0f && z <= s.max[2] + 64.0f) {
        site_zone_votes_[i][zone]++;
      }
    }
  }

  ReplayOptions options_;
  Replay* out_;
  std::vector<signed char> kind_by_class_;
  std::map<int, int> player_index_;      // controller entity -> players index
  std::map<int, PlayerSample> last_sample_;
  std::map<std::string, int> weapon_index_;
  std::set<int> sites_seen_;
  std::vector<std::map<int, int>> site_zone_votes_;
  bool have_bounds_ = false;
  float seen_min_[2] = {1e9f, 1e9f};
  float seen_max_[2] = {-1e9f, -1e9f};
};

}  // namespace

bool BuildReplay(DemoReader* reader, const ReplayOptions& options, Replay* out,
                 std::string* error) {
  *out = Replay();
  out->tick_step = options.tick_step > 0 ? options.tick_step : 1;

  EntityDecoder decoder;
  Builder builder(options, out);
  int next_sample = 0;
  std::string packet_error;
  int failures = 0;

  // The header names the map; everything else is entity state.
  DemoFrame first;
  if (!reader->Next(&first)) return false;
  if (first.kind == kDemFileHeader) {
    pb::Reader r(first.body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 5) out->map = r.ReadString();  // CDemoFileHeader.map_name
    }
  }

  const bool ok = WalkEntityFrames(reader, &decoder, [&](const DemoFrame& frame) {
    if (frame.kind != kDemPacket && frame.kind != kDemSignonPacket) return true;
    if (!ApplyPacketFrame(frame, &decoder, nullptr, &packet_error)) ++failures;
    if (frame.kind == kDemPacket && frame.tick >= next_sample) {
      builder.Sample(frame.tick, decoder);
      next_sample = (frame.tick / out->tick_step + 1) * out->tick_step;
    }
    return true;
  }, error);
  if (!ok) return false;
  builder.Finish();
  if (failures > 0 && out->frames.empty()) {
    if (error != nullptr) *error = "entity decoding failed: " + packet_error;
    return false;
  }
  return true;
}

std::string ReplayToJson(const Replay& replay, int from_tick, int to_tick) {
  JsonWriter w;
  w.BeginObject();
  w.Field("map", replay.map);
  w.Field("tickRate", replay.tick_rate);
  w.Field("tickStep", replay.tick_step);
  w.Key("bounds");
  w.BeginObject();
  w.Key("min");
  w.BeginArray();
  w.Fixed(replay.bounds_min[0], 1);
  w.Fixed(replay.bounds_min[1], 1);
  w.EndArray();
  w.Key("max");
  w.BeginArray();
  w.Fixed(replay.bounds_max[0], 1);
  w.Fixed(replay.bounds_max[1], 1);
  w.EndArray();
  w.EndObject();

  // The map image covers whole grid cells from the bounds' minimum corner,
  // so it can overhang the maximum edge by up to one cell.
  w.Key("grid");
  w.BeginObject();
  w.Field("cell", static_cast<double>(replay.grid_cell));
  w.Field("width", replay.grid_w);
  w.Field("height", replay.grid_h);
  w.EndObject();

  w.Key("sites");
  w.BeginArray();
  for (const Bombsite& s : replay.sites) {
    w.BeginObject();
    w.Field("letter", s.letter);
    w.Key("min");
    w.BeginArray();
    for (float v : s.min) w.Fixed(v, 1);
    w.EndArray();
    w.Key("max");
    w.BeginArray();
    for (float v : s.max) w.Fixed(v, 1);
    w.EndArray();
    w.EndObject();
  }
  w.EndArray();

  w.Key("players");
  w.BeginArray();
  for (const ReplayPlayer& p : replay.players) {
    w.BeginObject();
    w.Field("id", p.id);
    w.Field("name", p.name);
    w.FieldId("steamId", p.steam_id);
    w.EndObject();
  }
  w.EndArray();

  w.Key("weapons");
  w.BeginArray();
  for (const std::string& name : replay.weapons) w.Value(name);
  w.EndArray();

  // Frames as arrays: [tick, [[player, x, y, z, yaw, health, armor, team,
  // flags, weapon]...], [[kind, x, y, z, active, thrower]...], [bomb state,
  // x, y, z, carrier, site, defusing]].
  w.Key("frames");
  w.BeginArray();
  for (const ReplayFrame& frame : replay.frames) {
    if ((from_tick != 0 || to_tick != 0) && (frame.tick < from_tick || frame.tick > to_tick)) {
      continue;
    }
    w.BeginArray();
    w.Value(frame.tick);
    w.BeginArray();
    for (const PlayerSample& s : frame.players) {
      w.BeginArray();
      w.Value(s.player);
      w.Fixed(s.x, 1);
      w.Fixed(s.y, 1);
      w.Fixed(s.z, 1);
      w.Fixed(s.yaw, 1);
      w.Value(s.health);
      w.Value(s.armor);
      w.Value(s.team);
      w.Value(s.flags);
      w.Value(s.weapon);
      w.EndArray();
    }
    w.EndArray();
    w.BeginArray();
    for (const GrenadeSample& g : frame.grenades) {
      w.BeginArray();
      w.Value(g.kind);
      w.Fixed(g.x, 1);
      w.Fixed(g.y, 1);
      w.Fixed(g.z, 1);
      w.Value(g.active ? 1 : 0);
      w.Value(g.thrower);
      w.EndArray();
    }
    w.EndArray();
    w.BeginArray();
    w.Value(frame.bomb.state);
    w.Fixed(frame.bomb.x, 1);
    w.Fixed(frame.bomb.y, 1);
    w.Fixed(frame.bomb.z, 1);
    w.Value(frame.bomb.carrier);
    w.Value(frame.bomb.site);
    w.Value(frame.bomb.being_defused ? 1 : 0);
    w.EndArray();
    w.EndArray();
  }
  w.EndArray();
  w.EndObject();
  return w.str();
}

std::string ReplayWalkedPng(const Replay& replay, float z_min) {
  const int w = replay.grid_w;
  const int h = replay.grid_h;
  if (w <= 0 || h <= 0 || replay.visits.empty()) return std::string();
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(w) * h * 4, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // World y grows upwards; image rows grow downwards.
      const std::size_t at = static_cast<std::size_t>(h - 1 - y) * w + x;
      if (replay.visits[at] == 0) continue;
      if (at < replay.visit_top.size() && replay.visit_top[at] < z_min) continue;
      std::uint8_t* px = &pixels[(static_cast<std::size_t>(y) * w + x) * 4];
      px[0] = px[1] = px[2] = 255;
      px[3] = 255;
    }
  }
  return EncodePngRgba8(w, h, pixels);
}

std::string ReplayMapPng(const Replay& replay) {
  const int w = replay.grid_w;
  const int h = replay.grid_h;
  if (w <= 0 || h <= 0 || replay.visits.empty()) return std::string();
  const std::size_t n = static_cast<std::size_t>(w) * h;

  // Footsteps are sparse: a player covers a cell in a fraction of a second
  // and a corridor is walked along one line. Blurring the visit counts
  // turns those lines into the area they belong to, with soft edges; a
  // second, wider blur reaches into corners nobody stood in exactly. The
  // counts go in on a log scale so that spawns do not wash everything out.
  std::vector<float> density(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    if (replay.visits[i] > 0) density[i] = std::log1p(static_cast<float>(replay.visits[i]));
  }
  auto blur = [&](const std::vector<float>& in, int radius) {
    std::vector<float> tmp(n, 0.0f), out(n, 0.0f);
    const float norm = 1.0f / static_cast<float>(2 * radius + 1);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float sum = 0.0f;
        for (int d = -radius; d <= radius; ++d) {
          const int nx = x + d;
          if (nx >= 0 && nx < w) sum += in[static_cast<std::size_t>(y) * w + nx];
        }
        tmp[static_cast<std::size_t>(y) * w + x] = sum * norm;
      }
    }
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float sum = 0.0f;
        for (int d = -radius; d <= radius; ++d) {
          const int ny = y + d;
          if (ny >= 0 && ny < h) sum += tmp[static_cast<std::size_t>(ny) * w + x];
        }
        out[static_cast<std::size_t>(y) * w + x] = sum * norm;
      }
    }
    return out;
  };
  // A player is about 32 units across: one cell either side.
  const int reach = std::max(1, static_cast<int>(std::round(16.0f / replay.grid_cell)));
  const std::vector<float> near = blur(blur(density, reach), reach);
  const std::vector<float> wide = blur(near, reach * 3);

  float peak = 0.0f;
  for (float v : near) peak = std::max(peak, v);
  if (peak <= 0.0f) peak = 1.0f;

  // World y grows upwards; image rows grow downwards. Two tones: the area
  // players actually stood in, and a dimmer fringe around it that hints at
  // the rest of the walkable space.
  std::vector<std::uint8_t> pixels(n * 4, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(h - 1 - y) * w + x;
      std::uint8_t* px = &pixels[(static_cast<std::size_t>(y) * w + x) * 4];
      const float v = near[i] / peak;
      const float fringe = wide[i] / peak;
      if (v >= 0.015f) {
        // Most of the map is lightly walked next to the spawns, so the
        // shading saturates early and only the rarely visited edges fade.
        const float t = std::min(1.0f, v / 0.2f);
        const int grey = static_cast<int>(125.0f + 70.0f * t);
        px[0] = px[1] = px[2] = static_cast<std::uint8_t>(std::min(255, grey));
        px[3] = 255;
      } else if (fringe >= 0.003f) {
        px[0] = px[1] = px[2] = 90;
        px[3] = static_cast<std::uint8_t>(std::min(255.0f, 255.0f * std::min(1.0f, fringe / 0.015f)));
      }
    }
  }
  return EncodePngRgba8(w, h, pixels);
}

}  // namespace cs2mv
