#include "cs2mv/match.h"

#include <cmath>
#include <cstdio>

#include "cs2mv/json.h"

namespace cs2mv {
namespace {

// Re-indents compact JSON. Only used for the CLI's --pretty output, so it can
// afford to be simple; it just needs to leave string contents alone.
std::string Reindent(const std::string& in) {
  std::string out;
  out.reserve(in.size() * 2);
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const char c = in[i];
    if (in_string) {
      out += c;
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    switch (c) {
      case '"':
        in_string = true;
        out += c;
        break;
      case '{':
      case '[':
        out += c;
        // Keep empty containers on one line.
        if (i + 1 < in.size() && (in[i + 1] == '}' || in[i + 1] == ']')) break;
        ++depth;
        out += '\n';
        out.append(depth * 2, ' ');
        break;
      case '}':
      case ']':
        if (!out.empty() && out.back() != '{' && out.back() != '[') {
          --depth;
          out += '\n';
          out.append(depth * 2, ' ');
        }
        out += c;
        break;
      case ',':
        out += c;
        out += '\n';
        out.append(depth * 2, ' ');
        break;
      case ':':
        out += ": ";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

void WriteKill(JsonWriter* w, const Kill& k) {
  w->BeginObject();
  w->Field("tick", k.tick);
  w->Field("time", k.time);
  w->Field("attacker", k.attacker);
  w->Field("victim", k.victim);
  w->Field("assister", k.assister);
  w->Field("attackerTeam", k.attacker_team);
  w->Field("victimTeam", k.victim_team);
  w->Field("weapon", k.weapon);
  w->Field("headshot", k.headshot);
  w->Field("noscope", k.noscope);
  w->Field("throughSmoke", k.through_smoke);
  w->Field("attackerBlind", k.attacker_blind);
  w->Field("wallbang", k.wallbang);
  w->Field("assistedFlash", k.assist_flash);
  w->EndObject();
}

void WritePlayer(JsonWriter* w, const Player& p) {
  w->BeginObject();
  w->Field("slot", p.slot);
  w->Field("userId", p.user_id);
  w->FieldId("steamId", p.steam_id);
  w->Field("name", p.name);
  w->Field("team", p.team);
  w->Field("teamName", TeamName(p.team));
  w->Field("bot", p.bot);
  w->Field("hltv", p.hltv);
  w->Field("kills", p.kills);
  w->Field("deaths", p.deaths);
  w->Field("assists", p.assists);
  w->Field("headshots", p.headshots);
  w->Field("mvps", p.mvps);
  w->Field("damage", p.damage);
  w->Field("utilityDamage", p.utility_damage);
  w->Field("enemiesFlashed", p.enemies_flashed);
  w->Field("entryKills", p.entry_kills);
  w->Field("entryDeaths", p.entry_deaths);
  w->Field("roundsPlayed", p.rounds_played);
  w->Field("score", p.score);
  w->Field("kd", p.kd());
  w->Field("adr", p.adr());
  w->Field("hsPercent", p.hs_percent());
  w->EndObject();
}

void WriteRound(JsonWriter* w, const Round& r) {
  w->BeginObject();
  w->Field("number", r.number);
  w->Field("startTick", r.start_tick);
  w->Field("endTick", r.end_tick);
  w->Field("winner", r.winner);
  w->Field("winnerName", TeamName(r.winner));
  w->Field("reason", r.reason);
  w->Field("reasonText", r.reason_text);
  w->Field("winnerInferred", r.winner_inferred);
  w->Field("scoreT", r.score_t);
  w->Field("scoreCt", r.score_ct);
  w->Field("bombPlanted", r.bomb_planted);
  w->Field("bombDefused", r.bomb_defused);
  w->Field("bombExploded", r.bomb_exploded);
  w->Key("kills");
  w->BeginArray();
  for (const Kill& k : r.kills) WriteKill(w, k);
  w->EndArray();
  w->EndObject();
}

}  // namespace

const char* TeamName(int team) {
  switch (team) {
    case kTeamT: return "T";
    case kTeamCT: return "CT";
    case kTeamSpectator: return "SPEC";
    default: return "?";
  }
}

const char* RoundEndReasonName(int reason) {
  switch (reason) {
    case kReasonTargetBombed: return "Target bombed";
    case kReasonBombDefused: return "Bomb defused";
    case kReasonCTWin: return "Terrorists eliminated";
    case kReasonTWin: return "Counter-Terrorists eliminated";
    case kReasonTargetSaved: return "Time expired";
    default: return "";
  }
}

const Player* Match::PlayerBySlot(int slot) const {
  for (const Player& p : players) {
    if (p.slot == slot) return &p;
  }
  return nullptr;
}

std::string MatchToJson(const Match& m, bool pretty) {
  JsonWriter w;
  w.BeginObject();

  w.Key("match");
  w.BeginObject();
  w.Field("shareCode", m.share_code);
  w.FieldId("matchId", m.match_id);
  w.FieldId("outcomeId", m.outcome_id);
  w.Field("token", static_cast<long long>(m.token));
  w.Field("demoSource", m.demo_source);
  w.Field("map", m.map_name);
  w.Field("serverName", m.server_name);
  w.Field("clientName", m.client_name);
  w.Field("demoVersion", m.demo_version);
  w.Field("buildNumber", m.build_number);
  w.Field("tickInterval", m.tick_interval);
  w.Field("tickRate", m.tick_interval > 0.0 ? 1.0 / m.tick_interval : 0.0);
  w.Field("playbackTicks", m.playback_ticks);
  w.Field("playbackTime", m.playback_time);
  w.Field("scoreT", m.score_t);
  w.Field("scoreCt", m.score_ct);
  w.Field("roundsPlayed", m.rounds_played);
  w.EndObject();

  w.Key("players");
  w.BeginArray();
  for (const Player& p : m.players) WritePlayer(&w, p);
  w.EndArray();

  w.Key("rounds");
  w.BeginArray();
  for (const Round& r : m.rounds) WriteRound(&w, r);
  w.EndArray();

  w.Key("warnings");
  w.BeginArray();
  for (const std::string& s : m.warnings) w.Value(s);
  w.EndArray();

  w.EndObject();
  return pretty ? Reindent(w.str()) : w.str();
}

}  // namespace cs2mv
