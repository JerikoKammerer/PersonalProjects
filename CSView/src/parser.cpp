#include "cs2mv/parser.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "cs2mv/bits.h"
#include "cs2mv/protobuf.h"

namespace cs2mv {
namespace {

// Message kinds carried inside a CDemoPacket. Values come from Valve's
// netmessages.proto (SVC_Messages) and gameevents.proto (EBaseGameEvents).
enum MessageKind {
  kNetTick = 4,
  kSvcServerInfo = 40,
  kSvcCreateStringTable = 44,
  kSvcUpdateStringTable = 45,
  kGeSource1LegacyGameEventList = 205,
  kGeSource1LegacyGameEvent = 207,
};

// A single value out of a CMsgSource1LegacyGameEvent.key_t. Exactly one of the
// val_* fields is set; rather than trusting the companion `type` field this
// records which one actually appeared.
struct EventValue {
  enum Kind { kNone, kString, kFloat, kInt, kBool, kUInt64 };
  Kind kind = kNone;
  std::string s;
  double f = 0.0;
  long long i = 0;
  bool b = false;
  unsigned long long u = 0;

  long long AsInt() const {
    switch (kind) {
      case kInt: return i;
      case kBool: return b ? 1 : 0;
      case kFloat: return static_cast<long long>(f);
      case kUInt64: return static_cast<long long>(u);
      default: return 0;
    }
  }
  double AsFloat() const { return kind == kFloat ? f : static_cast<double>(AsInt()); }
  bool AsBool() const { return kind == kBool ? b : AsInt() != 0; }
};

struct EventDescriptor {
  std::string name;
  std::vector<std::string> keys;
};

// One decoded game event: the descriptor's key names paired with the values
// that arrived, in order.
class EventArgs {
 public:
  void Add(const std::string* name, EventValue v) {
    keys_.emplace_back(name, std::move(v));
  }

  const EventValue* Find(const char* name) const {
    for (const auto& kv : keys_) {
      if (kv.first != nullptr && *kv.first == name) return &kv.second;
    }
    return nullptr;
  }

  long long Int(const char* name, long long fallback = 0) const {
    const EventValue* v = Find(name);
    return v != nullptr ? v->AsInt() : fallback;
  }
  double Float(const char* name, double fallback = 0.0) const {
    const EventValue* v = Find(name);
    return v != nullptr ? v->AsFloat() : fallback;
  }
  bool Bool(const char* name, bool fallback = false) const {
    const EventValue* v = Find(name);
    return v != nullptr ? v->AsBool() : fallback;
  }
  std::string Str(const char* name) const {
    const EventValue* v = Find(name);
    return (v != nullptr && v->kind == EventValue::kString) ? v->s : std::string();
  }
  bool Has(const char* name) const { return Find(name) != nullptr; }

 private:
  std::vector<std::pair<const std::string*, EventValue>> keys_;
};

bool IsUtilityWeapon(const std::string& w) {
  return w == "hegrenade" || w == "inferno" || w == "molotov" ||
         w == "incgrenade" || w == "flashbang" || w == "decoy" ||
         w == "smokegrenade" || w == "firebomb" || w == "molotov_projectile";
}

// Reads the `data` field out of a CDemoPacket (field 3). Falls back to the
// first length-delimited field so that a future field renumbering degrades to
// "wrong data" rather than "no data".
pb::Slice PacketPayload(const std::string& body) {
  pb::Reader r(body);
  pb::Slice found;
  pb::Slice first;
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (r.wire_type() != pb::kLengthDelimited) continue;
    const pb::Slice s = r.ReadBytes();
    if (field == 3) {
      found = s;
      break;
    }
    if (first.data == nullptr) first = s;
  }
  return found.data != nullptr ? found : first;
}

// Tallies the table names in a CDemoStringTables { tables = 1 repeated
// table_t { table_name = 1, ... } }, for the inventory report.
void CountStringTableNames(const pb::Slice& tables, DemoInventory* out) {
  pb::Reader r(tables);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (field != 1 || r.wire_type() != pb::kLengthDelimited) continue;
    pb::Reader table(r.ReadBytes());
    std::uint32_t table_field = 0;
    while (table.NextField(&table_field)) {
      if (table_field == 1 && table.wire_type() == pb::kLengthDelimited) {
        out->string_tables[table.ReadString()]++;
      }
    }
  }
}

class MatchParser {
 public:
  MatchParser(const ParseOptions& options, Match* match)
      : options_(options), match_(match) {}

  bool Run(DemoReader* reader, std::string* error) {
    DemoFrame frame;
    while (reader->Next(&frame)) {
      HandleFrame(frame);
    }
    if (reader->failed()) {
      // A truncated demo still yields everything parsed up to the break, so
      // this is reported as a warning and the match is kept.
      Warn("demo ended early: " + reader->error());
    }
    Finish();
    (void)error;
    return true;
  }

 private:
  void Warn(const std::string& msg) {
    if (static_cast<int>(match_->warnings.size()) >= options_.max_warnings) return;
    match_->warnings.push_back(msg);
  }

  // ---------------------------------------------------------------- frames

  void HandleFrame(const DemoFrame& frame) {
    switch (frame.kind) {
      case kDemFileHeader:
        HandleFileHeader(frame.body);
        break;
      case kDemFileInfo:
        HandleFileInfo(frame.body);
        break;
      case kDemStringTables:
        HandleStringTables(pb::Reader(frame.body));
        break;
      case kDemFullPacket:
        // The embedded packet duplicates messages that also arrive as
        // DEM_Packet frames, so only the string table snapshot is taken.
        HandleFullPacket(frame.body);
        break;
      case kDemPacket:
      case kDemSignonPacket:
        tick_ = frame.tick;
        HandlePacket(frame.body);
        break;
      default:
        break;
    }
  }

  // CDemoFileHeader
  void HandleFileHeader(const std::string& body) {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 3: match_->server_name = r.ReadString(); break;
        case 4: match_->client_name = r.ReadString(); break;
        case 5:
          if (match_->map_name.empty()) match_->map_name = r.ReadString();
          break;
        case 11: match_->demo_version = r.ReadString(); break;
        case 13: match_->build_number = r.ReadInt32(); break;
        default: break;
      }
    }
  }

  // CDemoFileInfo
  void HandleFileInfo(const std::string& body) {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 1: match_->playback_time = r.ReadFloat(); break;
        case 2: match_->playback_ticks = r.ReadInt32(); break;
        default: break;
      }
    }
  }

  // CDemoFullPacket { string_table = 1, packet = 2 }
  void HandleFullPacket(const std::string& body) {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 1 && r.wire_type() == pb::kLengthDelimited) {
        HandleStringTables(pb::Reader(r.ReadBytes()));
      }
    }
  }

  // CDemoStringTables { tables = 1 repeated table_t }
  void HandleStringTables(pb::Reader r) {
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field != 1 || r.wire_type() != pb::kLengthDelimited) continue;
      const pb::Slice table = r.ReadBytes();
      HandleStringTable(table);
    }
  }

  // CDemoStringTables.table_t { table_name = 1, items = 2 repeated items_t }
  void HandleStringTable(const pb::Slice& table) {
    pb::Reader r(table);
    std::string name;
    std::vector<pb::Slice> items;
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 1) {
        name = r.ReadString();
      } else if (field == 2 && r.wire_type() == pb::kLengthDelimited) {
        items.push_back(r.ReadBytes());
      }
    }
    if (name != "userinfo") return;
    for (std::size_t i = 0; i < items.size(); ++i) {
      HandleUserInfoEntry(static_cast<int>(i), items[i]);
    }
  }

  // CDemoStringTables.items_t { str = 1, data = 2 }, where data holds a
  // serialised CMsgPlayerInfo for the userinfo table.
  void HandleUserInfoEntry(int slot, const pb::Slice& item) {
    pb::Reader r(item);
    pb::Slice data;
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 2 && r.wire_type() == pb::kLengthDelimited) data = r.ReadBytes();
    }
    if (data.data == nullptr || data.size == 0) return;

    // CMsgPlayerInfo { name = 1, xuid = 2 fixed64, userid = 3, steamid = 4
    //                  fixed64, fakeplayer = 5, ishltv = 6 }
    pb::Reader p(data);
    std::string name;
    std::uint64_t xuid = 0;
    std::uint64_t steamid = 0;
    long long userid = -1;
    bool fake = false;
    bool hltv = false;
    while (p.NextField(&field)) {
      switch (field) {
        case 1: name = p.ReadString(); break;
        case 2: xuid = p.ReadFixed64(); break;
        case 3: userid = p.ReadInt32(); break;
        case 4: steamid = p.ReadFixed64(); break;
        case 5: fake = p.ReadBool(); break;
        case 6: hltv = p.ReadBool(); break;
        default: break;
      }
    }
    if (!p.ok()) return;
    if (name.empty() && xuid == 0 && steamid == 0) return;  // empty slot

    const int idx = Upsert(slot, userid, xuid != 0 ? xuid : steamid, name);
    if (idx < 0) return;
    Player& pl = match_->players[static_cast<std::size_t>(idx)];
    pl.bot = pl.bot || fake;
    pl.hltv = pl.hltv || hltv;
  }

  // ------------------------------------------------------------- messages

  void HandlePacket(const std::string& body) {
    const pb::Slice payload = PacketPayload(body);
    if (payload.data == nullptr) return;

    BitReader bits(payload.data, payload.size);
    std::string buf;
    while (bits.BitsLeft() > 8) {
      const std::uint32_t kind = bits.ReadUBitVar();
      const std::uint32_t size = bits.ReadVarUInt32();
      if (!bits.ok()) break;
      if (size > (1u << 24)) {  // no legitimate message is 16 MB
        Warn("packet message claims an implausible size, skipping rest of frame");
        break;
      }
      buf.resize(size);
      if (size > 0 && !bits.ReadBytes(&buf[0], size)) break;
      HandleMessage(kind, buf);
    }
  }

  void HandleMessage(std::uint32_t kind, const std::string& body) {
    switch (kind) {
      case kSvcServerInfo:
        HandleServerInfo(body);
        break;
      case kGeSource1LegacyGameEventList:
        HandleGameEventList(body);
        break;
      case kGeSource1LegacyGameEvent:
        HandleGameEvent(body);
        break;
      default:
        break;
    }
  }

  // CSVCMsg_ServerInfo { max_clients = 10, tick_interval = 13, game_dir = 14,
  //                      map_name = 15, host_name = 17 }
  void HandleServerInfo(const std::string& body) {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 13: {
          const double interval = r.ReadFloat();
          if (interval > 0.0 && interval < 1.0) match_->tick_interval = interval;
          break;
        }
        case 15: {
          const std::string map = r.ReadString();
          if (!map.empty()) match_->map_name = map;
          break;
        }
        case 17: {
          const std::string host = r.ReadString();
          if (match_->server_name.empty()) match_->server_name = host;
          break;
        }
        default:
          break;
      }
    }
  }

  // CMsgSource1LegacyGameEventList { descriptors = 1 }
  void HandleGameEventList(const std::string& body) {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field != 1 || r.wire_type() != pb::kLengthDelimited) continue;
      const pb::Slice d = r.ReadBytes();

      // descriptor_t { eventid = 1, name = 2, keys = 3 }
      pb::Reader dr(d);
      int event_id = -1;
      EventDescriptor desc;
      std::uint32_t f = 0;
      while (dr.NextField(&f)) {
        switch (f) {
          case 1: event_id = dr.ReadInt32(); break;
          case 2: desc.name = dr.ReadString(); break;
          case 3: {
            // key_t { type = 1, name = 2 }
            const pb::Slice k = dr.ReadBytes();
            pb::Reader kr(k);
            std::string key_name;
            std::uint32_t kf = 0;
            while (kr.NextField(&kf)) {
              if (kf == 2) key_name = kr.ReadString();
            }
            desc.keys.push_back(std::move(key_name));
            break;
          }
          default:
            break;
        }
      }
      if (event_id >= 0) descriptors_[event_id] = std::move(desc);
    }
  }

  // CMsgSource1LegacyGameEvent { event_name = 1, eventid = 2, keys = 3 }
  void HandleGameEvent(const std::string& body) {
    pb::Reader r(body);
    std::string inline_name;
    int event_id = -1;
    std::vector<EventValue> values;
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 1: inline_name = r.ReadString(); break;
        case 2: event_id = r.ReadInt32(); break;
        case 3: {
          const pb::Slice k = r.ReadBytes();
          values.push_back(ParseEventValue(k));
          break;
        }
        default:
          break;
      }
    }
    if (!r.ok()) return;

    const EventDescriptor* desc = nullptr;
    auto it = descriptors_.find(event_id);
    if (it != descriptors_.end()) desc = &it->second;

    const std::string& name = (desc != nullptr && !desc->name.empty()) ? desc->name : inline_name;
    if (name.empty()) return;

    EventArgs args;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::string* key_name = nullptr;
      if (desc != nullptr && i < desc->keys.size()) key_name = &desc->keys[i];
      args.Add(key_name, std::move(values[i]));
    }
    Dispatch(name, args);
  }

  // CMsgSource1LegacyGameEvent.key_t
  static EventValue ParseEventValue(const pb::Slice& k) {
    EventValue v;
    pb::Reader r(k);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      switch (field) {
        case 2: v.kind = EventValue::kString; v.s = r.ReadString(); break;
        case 3: v.kind = EventValue::kFloat; v.f = r.ReadFloat(); break;
        case 4: v.kind = EventValue::kInt; v.i = r.ReadInt32(); break;
        case 5: v.kind = EventValue::kInt; v.i = r.ReadInt32(); break;
        case 6: v.kind = EventValue::kInt; v.i = r.ReadInt32(); break;
        case 7: v.kind = EventValue::kBool; v.b = r.ReadBool(); break;
        case 8: v.kind = EventValue::kUInt64; v.u = r.ReadVarint(); break;
        default: break;  // field 1 is the type tag, which is redundant here
      }
    }
    return v;
  }

  // --------------------------------------------------------------- events

  void Dispatch(const std::string& name, const EventArgs& args) {
    if (name == "player_death") return OnPlayerDeath(args);
    if (name == "player_hurt") return OnPlayerHurt(args);
    if (name == "player_blind") return OnPlayerBlind(args);
    if (name == "player_team") return OnPlayerTeam(args);
    if (name == "player_connect") return OnPlayerConnect(args);
    if (name == "player_disconnect") return OnPlayerConnect(args);  // same keys
    if (name == "round_start") return OnRoundStart(args);
    if (name == "round_end") return OnRoundEnd(args);
    if (name == "round_mvp") return OnRoundMvp(args);
    if (name == "bomb_planted") return OnBomb(1);
    if (name == "bomb_defused") return OnBomb(2);
    if (name == "bomb_exploded") return OnBomb(3);
    if (name == "begin_new_match" || name == "round_announce_match_start") {
      return OnMatchStart();
    }
  }

  void OnPlayerConnect(const EventArgs& args) {
    const std::string name = args.Str("name");
    const long long userid = args.Has("userid") ? args.Int("userid", -1) : -1;
    std::uint64_t xuid = 0;
    if (const EventValue* v = args.Find("xuid")) {
      xuid = v->kind == EventValue::kUInt64 ? v->u
                                            : static_cast<std::uint64_t>(v->AsInt());
    }
    if (name.empty() && xuid == 0) return;
    const int idx = Upsert(-1, userid, xuid, name);
    if (idx >= 0 && args.Bool("bot", false)) {
      match_->players[static_cast<std::size_t>(idx)].bot = true;
    }
  }

  void OnPlayerTeam(const EventArgs& args) {
    const int idx = PlayerFromEvent(args, "userid");
    if (idx < 0) return;
    // Team 0 means "unassigned", which CS2 also sends on disconnect. Keeping
    // the last real side means a player who left still shows up on the team
    // they played for instead of dropping to the bottom of the scoreboard.
    const int team = static_cast<int>(args.Int("team", kTeamUnknown));
    if (team >= kTeamSpectator && team <= kTeamCT) {
      match_->players[static_cast<std::size_t>(idx)].team = team;
    }
  }

  void OnPlayerDeath(const EventArgs& args) {
    const int victim = PlayerFromEvent(args, "userid");
    const int attacker = PlayerFromEvent(args, "attacker");
    const int assister = PlayerFromEvent(args, "assister");

    Kill k;
    k.tick = tick_;
    k.time = round_start_tick_ > 0
                 ? (tick_ - round_start_tick_) * match_->tick_interval
                 : 0.0;
    k.victim = victim;
    k.attacker = attacker;
    k.assister = assister;
    k.weapon = args.Str("weapon");
    k.headshot = args.Bool("headshot");
    k.noscope = args.Bool("noscope");
    k.through_smoke = args.Bool("thrusmoke");
    k.attacker_blind = args.Bool("attackerblind");
    k.wallbang = args.Int("penetrated", 0) > 0;
    k.assist_flash = args.Bool("assistedflash");
    k.attacker_team = TeamOf(attacker);
    k.victim_team = TeamOf(victim);
    current_.kills.push_back(k);

    if (victim >= 0) match_->players[static_cast<std::size_t>(victim)].deaths++;
    if (attacker >= 0 && attacker != victim) {
      Player& a = match_->players[static_cast<std::size_t>(attacker)];
      // Team kills subtract, matching the in-game scoreboard.
      if (k.attacker_team != kTeamUnknown && k.attacker_team == k.victim_team) {
        a.kills--;
      } else {
        a.kills++;
        if (k.headshot) a.headshots++;
      }
    }
    if (assister >= 0 && assister != victim) {
      match_->players[static_cast<std::size_t>(assister)].assists++;
    }
  }

  void OnPlayerHurt(const EventArgs& args) {
    const int victim = PlayerFromEvent(args, "userid");
    const int attacker = PlayerFromEvent(args, "attacker");
    if (attacker < 0 || attacker == victim) return;
    const int team_a = TeamOf(attacker);
    const int team_v = TeamOf(victim);
    if (team_a != kTeamUnknown && team_a == team_v) return;  // team damage

    const int dmg = static_cast<int>(args.Int("dmg_health", 0));
    if (dmg <= 0) return;
    Player& a = match_->players[static_cast<std::size_t>(attacker)];
    a.damage += dmg;
    if (IsUtilityWeapon(args.Str("weapon"))) a.utility_damage += dmg;
  }

  void OnPlayerBlind(const EventArgs& args) {
    const int victim = PlayerFromEvent(args, "userid");
    const int attacker = PlayerFromEvent(args, "attacker");
    if (attacker < 0 || attacker == victim) return;
    const int team_a = TeamOf(attacker);
    const int team_v = TeamOf(victim);
    if (team_a != kTeamUnknown && team_a == team_v) return;
    // The usual threshold for a flash that actually did something.
    if (args.Float("blind_duration", 0.0) < 0.7) return;
    match_->players[static_cast<std::size_t>(attacker)].enemies_flashed++;
  }

  void OnRoundStart(const EventArgs&) {
    Flush();
    current_ = Round();
    current_.start_tick = tick_;
    round_start_tick_ = tick_;
  }

  void OnRoundEnd(const EventArgs& args) {
    const int winner = static_cast<int>(args.Int("winner", kTeamUnknown));
    current_.winner = winner;
    current_.reason = static_cast<int>(args.Int("reason", 0));
    current_.reason_text = RoundEndReasonName(current_.reason);
    if (current_.reason_text.empty()) current_.reason_text = args.Str("message");
    current_.end_tick = tick_;
  }

  void OnRoundMvp(const EventArgs& args) {
    const int idx = PlayerFromEvent(args, "userid");
    if (idx >= 0) match_->players[static_cast<std::size_t>(idx)].mvps++;
  }

  void OnBomb(int what) {
    if (what == 1) current_.bomb_planted = true;
    if (what == 2) current_.bomb_defused = true;
    if (what == 3) current_.bomb_exploded = true;
  }

  // Warmup, knife rounds and map vetoes all happen before this fires.
  void OnMatchStart() {
    if (!options_.reset_on_match_start) return;
    if (match_started_) return;
    match_started_ = true;
    match_->rounds.clear();
    match_->score_t = 0;
    match_->score_ct = 0;
    for (Player& p : match_->players) {
      const int team = p.team;
      Player fresh;
      fresh.slot = p.slot;
      fresh.user_id = p.user_id;
      fresh.steam_id = p.steam_id;
      fresh.name = p.name;
      fresh.bot = p.bot;
      fresh.hltv = p.hltv;
      fresh.team = team;
      p = fresh;
    }
    current_ = Round();
    current_.start_tick = tick_;
    round_start_tick_ = tick_;
  }

  // Commits the round being accumulated, if it actually finished.
  void Flush() {
    if (current_.winner != kTeamT && current_.winner != kTeamCT) {
      // Warmup and aborted rounds never get a winner; drop them.
      current_ = Round();
      return;
    }
    if (current_.winner == kTeamT) {
      match_->score_t++;
    } else {
      match_->score_ct++;
    }
    current_.number = static_cast<int>(match_->rounds.size()) + 1;
    current_.score_t = match_->score_t;
    current_.score_ct = match_->score_ct;

    if (!current_.kills.empty()) {
      const Kill& first = current_.kills.front();
      if (first.attacker >= 0) {
        match_->players[static_cast<std::size_t>(first.attacker)].entry_kills++;
      }
      if (first.victim >= 0) {
        match_->players[static_cast<std::size_t>(first.victim)].entry_deaths++;
      }
    }
    for (Player& p : match_->players) {
      if (p.team == kTeamT || p.team == kTeamCT) p.rounds_played++;
    }

    match_->rounds.push_back(current_);
    current_ = Round();
  }

  void Finish() {
    Flush();
    match_->rounds_played = static_cast<int>(match_->rounds.size());
    if (match_->playback_ticks > 0 && match_->playback_time <= 0.0) {
      match_->playback_time = match_->playback_ticks * match_->tick_interval;
    }

    // A demo whose events reference players in a way this parser does not
    // recognise would otherwise produce a plausible looking but empty
    // scoreboard, so say so out loud.
    if (resolve_attempts_ > 20 && resolve_failures_ * 2 > resolve_attempts_) {
      Warn("could not match " + std::to_string(resolve_failures_) + " of " +
           std::to_string(resolve_attempts_) +
           " player references in game events to a player; the scoreboard is "
           "incomplete. See PlayerFromEvent in src/parser.cpp.");
    }
    if (match_->players.empty()) {
      Warn("no players found: the demo carried no userinfo string table and no "
           "player_connect events");
    }
    SortPlayers();
  }

  // Orders the scoreboard the way the game does: by team, then kills, with the
  // usual tiebreakers. Kills reference players by index, so the permutation is
  // applied to them too.
  void SortPlayers() {
    const std::size_t n = match_->players.size();
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = static_cast<int>(i);

    const std::vector<Player>& players = match_->players;
    std::stable_sort(order.begin(), order.end(), [&players](int a, int b) {
      const Player& pa = players[static_cast<std::size_t>(a)];
      const Player& pb_ = players[static_cast<std::size_t>(b)];
      if (pa.hltv != pb_.hltv) return !pa.hltv;      // observers last
      if (pa.team != pb_.team) return pa.team > pb_.team;  // CT (3) before T (2)
      if (pa.kills != pb_.kills) return pa.kills > pb_.kills;
      if (pa.deaths != pb_.deaths) return pa.deaths < pb_.deaths;
      return pa.name < pb_.name;
    });

    std::vector<int> remap(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
      remap[static_cast<std::size_t>(order[i])] = static_cast<int>(i);
    }
    std::vector<Player> sorted;
    sorted.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      sorted.push_back(match_->players[static_cast<std::size_t>(order[i])]);
    }
    match_->players.swap(sorted);

    auto fix = [&remap, n](int* idx) {
      if (*idx < 0 || static_cast<std::size_t>(*idx) >= n) {
        *idx = -1;
      } else {
        *idx = remap[static_cast<std::size_t>(*idx)];
      }
    };
    for (Round& r : match_->rounds) {
      for (Kill& k : r.kills) {
        fix(&k.attacker);
        fix(&k.victim);
        fix(&k.assister);
      }
    }
  }

  // -------------------------------------------------------------- players

  int TeamOf(int idx) const {
    if (idx < 0) return kTeamUnknown;
    return match_->players[static_cast<std::size_t>(idx)].team;
  }

  // Resolves a game event's player reference.
  //
  // CS2 packs a generation counter into the bits above the low byte of a user
  // id, so the low byte is what identifies the player; this mirrors what
  // demoinfocs-golang does for Source 2 demos. Zero is the engine's "nobody"
  // (world damage, a death with no killer, an absent assister) and never a
  // player.
  //
  // The user id map is authoritative. The slot fallback only runs when no user
  // ids were ever learned, because slots and user ids share a number space and
  // guessing between them would silently credit the wrong player.
  int PlayerFromEvent(const EventArgs& args, const char* key) {
    const EventValue* v = args.Find(key);
    if (v == nullptr) return -1;
    long long raw = v->AsInt();
    if (raw <= 0) return -1;
    if (raw <= 0xFFFF) raw &= 0xFF;
    if (raw == 0) return -1;

    ++resolve_attempts_;
    auto by_user = by_user_id_.find(raw);
    if (by_user != by_user_id_.end()) return by_user->second;
    if (by_user_id_.empty()) {
      auto by_slot = by_slot_.find(static_cast<int>(raw));
      if (by_slot != by_slot_.end()) return by_slot->second;
    }
    ++resolve_failures_;
    return -1;
  }

  // Creates or updates the player record identified by any of slot, user id or
  // steam id, merging the userinfo string table view with the one that game
  // events give.
  int Upsert(int slot, long long user_id, std::uint64_t steam_id,
             const std::string& name) {
    int idx = -1;
    if (steam_id != 0) {
      auto it = by_steam_id_.find(steam_id);
      if (it != by_steam_id_.end()) idx = it->second;
    }
    if (idx < 0 && slot >= 0) {
      auto it = by_slot_.find(slot);
      if (it != by_slot_.end()) idx = it->second;
    }
    if (idx < 0 && user_id >= 0) {
      auto it = by_user_id_.find(user_id);
      if (it != by_user_id_.end()) idx = it->second;
    }
    if (idx < 0) {
      idx = static_cast<int>(match_->players.size());
      match_->players.push_back(Player());
    }

    Player& p = match_->players[static_cast<std::size_t>(idx)];
    if (slot >= 0) p.slot = slot;
    if (user_id >= 0) p.user_id = static_cast<int>(user_id);
    if (steam_id != 0) p.steam_id = steam_id;
    if (!name.empty()) p.name = name;

    if (slot >= 0) by_slot_[slot] = idx;
    if (user_id >= 0) by_user_id_[user_id] = idx;
    if (steam_id != 0) by_steam_id_[steam_id] = idx;
    return idx;
  }

  const ParseOptions& options_;
  Match* match_;

  std::map<int, EventDescriptor> descriptors_;
  std::map<long long, int> by_user_id_;
  std::map<int, int> by_slot_;
  std::map<std::uint64_t, int> by_steam_id_;

  Round current_;
  std::int32_t tick_ = 0;
  std::int32_t round_start_tick_ = 0;
  bool match_started_ = false;
  long long resolve_attempts_ = 0;
  long long resolve_failures_ = 0;
};

}  // namespace

bool ParseDemo(DemoReader* reader, const ParseOptions& options, Match* out,
               std::string* error) {
  MatchParser parser(options, out);
  return parser.Run(reader, error);
}

bool InspectDemo(DemoReader* reader, DemoInventory* out, std::string* error) {
  std::map<int, EventDescriptor> descriptors;
  DemoFrame frame;
  while (reader->Next(&frame)) {
    out->frames[frame.kind]++;
    out->total_bytes += static_cast<long long>(frame.body.size());
    if (frame.compressed) out->compressed_frames++;
    out->last_tick = frame.tick;

    if (frame.kind == kDemStringTables || frame.kind == kDemFullPacket) {
      // Only the table names are of interest here. A DEM_StringTables body is
      // a CDemoStringTables; a DEM_FullPacket body wraps one in field 1, so
      // the two kinds start at different depths.
      if (frame.kind == kDemStringTables) {
        CountStringTableNames(pb::Slice{
            reinterpret_cast<const std::uint8_t*>(frame.body.data()),
            frame.body.size()}, out);
      } else {
        pb::Reader r(frame.body);
        std::uint32_t field = 0;
        while (r.NextField(&field)) {
          if (field == 1 && r.wire_type() == pb::kLengthDelimited) {
            CountStringTableNames(r.ReadBytes(), out);
          }
        }
      }
    }

    if (frame.kind != kDemPacket && frame.kind != kDemSignonPacket) continue;
    const pb::Slice payload = PacketPayload(frame.body);
    if (payload.data == nullptr) continue;

    BitReader bits(payload.data, payload.size);
    std::string buf;
    while (bits.BitsLeft() > 8) {
      const std::uint32_t kind = bits.ReadUBitVar();
      const std::uint32_t size = bits.ReadVarUInt32();
      if (!bits.ok() || size > (1u << 24)) break;
      buf.resize(size);
      if (size > 0 && !bits.ReadBytes(&buf[0], size)) break;
      out->messages[static_cast<int>(kind)]++;

      if (kind == kGeSource1LegacyGameEventList) {
        pb::Reader r(buf);
        std::uint32_t field = 0;
        while (r.NextField(&field)) {
          if (field != 1 || r.wire_type() != pb::kLengthDelimited) continue;
          const pb::Slice d = r.ReadBytes();
          pb::Reader dr(d);
          int id = -1;
          EventDescriptor desc;
          std::uint32_t f = 0;
          while (dr.NextField(&f)) {
            if (f == 1) id = dr.ReadInt32();
            else if (f == 2) desc.name = dr.ReadString();
          }
          if (id >= 0) descriptors[id] = std::move(desc);
        }
      } else if (kind == kGeSource1LegacyGameEvent) {
        pb::Reader r(buf);
        std::uint32_t field = 0;
        std::string name;
        int id = -1;
        while (r.NextField(&field)) {
          if (field == 1) name = r.ReadString();
          else if (field == 2) id = r.ReadInt32();
        }
        auto it = descriptors.find(id);
        if (it != descriptors.end() && !it->second.name.empty()) name = it->second.name;
        if (name.empty()) name = "event#" + std::to_string(id);
        out->events[name]++;
      }
    }
  }
  if (reader->failed()) {
    if (error != nullptr) *error = reader->error();
    return false;
  }
  return true;
}

}  // namespace cs2mv
