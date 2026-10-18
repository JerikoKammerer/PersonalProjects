#include "cs2mv/publish.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include "cs2mv/json.h"

namespace cs2mv {
namespace {

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

bool WriteFile(const std::string& path, const std::string& body, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return Err(error, "cannot write " + path);
  out.write(body.data(), static_cast<std::streamsize>(body.size()));
  if (!out) return Err(error, "failed while writing " + path);
  return true;
}

// A published match is addressed by its id in a URL and by its name on disk,
// so anything that is awkward in either becomes an underscore.
std::string SafeId(const std::string& raw) {
  std::string out;
  for (char c : raw) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_';
    out.push_back(ok ? c : '_');
  }
  if (out.empty()) out = "match";
  return out;
}

// The demo's modification time, as a unix timestamp. C++17's file clock has
// no portable epoch, so it is anchored against its own now().
long long FileTime(const std::string& path) {
  std::error_code ec;
  const auto written = std::filesystem::last_write_time(path, ec);
  if (ec) return 0;
  const auto now_file = std::filesystem::file_time_type::clock::now();
  const auto now_sys = std::chrono::system_clock::now();
  const auto delta = std::chrono::duration_cast<std::chrono::seconds>(written - now_file);
  return std::chrono::duration_cast<std::chrono::seconds>(
             (now_sys + delta).time_since_epoch())
      .count();
}

}  // namespace

bool PublishMatch(const Match& match, const Replay& replay,
                  const std::string& out_dir, const std::string& demo_path,
                  std::string* id, std::string* error) {
  *id = SafeId(!match.share_code.empty() ? match.share_code
                                         : "match_" + std::to_string(match.match_id));

  std::error_code ec;
  const std::filesystem::path dir = std::filesystem::path(out_dir) / *id;
  std::filesystem::create_directories(dir, ec);
  if (ec) return Err(error, "cannot create " + dir.string() + ": " + ec.message());

  if (!WriteFile((dir / "match.json").string(), MatchToJson(match, false), error)) {
    return false;
  }
  const std::string map_png = ReplayMapPng(replay);
  if (!map_png.empty() && !WriteFile((dir / "map.png").string(), map_png, error)) {
    return false;
  }
  const std::string walked_png = ReplayWalkedPng(replay);
  if (!walked_png.empty() && !WriteFile((dir / "walked.png").string(), walked_png, error)) {
    return false;
  }
  // Nuke's drawing shows one floor, so the page asks for a height-filtered
  // footprint. Which height it asks for is the page's business, so publish
  // the floors it might want rather than guessing here.
  const std::string upper_png = ReplayWalkedPng(replay, -495.0f);
  if (!upper_png.empty() &&
      !WriteFile((dir / "walked-upper.png").string(), upper_png, error)) {
    return false;
  }

  // One file per round, which is what the page asks for anyway: a whole
  // match in one file is tens of megabytes.
  for (const Round& round : match.rounds) {
    const std::string name = "round-" + std::to_string(round.number) + ".json";
    if (!WriteFile((dir / name).string(),
                   ReplayToJson(replay, round.start_tick, round.end_tick), error)) {
      return false;
    }
  }

  // The catalogue entry. It is written whole so that index.json can be built
  // by concatenating these without parsing any JSON back in.
  const long long date = FileTime(demo_path);
  JsonWriter w;
  w.BeginObject();
  w.Field("id", *id);
  w.Field("shareCode", match.share_code);
  w.Field("map", match.map_name);
  w.Field("date", date);
  w.Field("scoreCt", match.score_ct);
  w.Field("scoreT", match.score_t);
  w.Field("rounds", match.rounds_played);
  w.Field("tickRate", match.tick_interval > 0.0 ? 1.0 / match.tick_interval : 0.0);
  w.Key("players");
  w.BeginArray();
  for (const Player& p : match.players) {
    if (p.hltv) continue;
    w.BeginObject();
    w.Field("name", p.name);
    w.FieldId("steamId", p.steam_id);
    w.Field("team", p.team);
    w.Field("kills", p.kills);
    w.Field("deaths", p.deaths);
    w.EndObject();
  }
  w.EndArray();
  w.EndObject();
  if (!WriteFile((dir / "card.json").string(), w.str(), error)) return false;
  // The sort key, kept beside the card so the catalogue can order matches
  // without reading JSON.
  return WriteFile((dir / "card.key").string(), std::to_string(date), error);
}

bool WriteCatalog(const std::string& out_dir, std::string* error) {
  struct Entry {
    long long date = 0;
    std::string json;
  };
  std::vector<Entry> entries;

  std::error_code ec;
  std::filesystem::directory_iterator it(out_dir, ec), end;
  if (ec) return Err(error, "cannot read " + out_dir + ": " + ec.message());
  for (; it != end; it.increment(ec)) {
    if (ec) break;
    if (!it->is_directory()) continue;
    const std::filesystem::path card = it->path() / "card.json";
    std::ifstream in(card, std::ios::binary);
    if (!in) continue;
    Entry entry;
    entry.json.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (entry.json.empty()) continue;
    std::ifstream key(it->path() / "card.key");
    if (key) key >> entry.date;
    entries.push_back(std::move(entry));
  }
  // Newest first, which is how the page lists them.
  std::sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) { return a.date > b.date; });

  std::string body = "{\"matches\":[";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i != 0) body += ',';
    body += entries[i].json;
  }
  body += "]}";
  return WriteFile((std::filesystem::path(out_dir) / "index.json").string(), body, error);
}

}  // namespace cs2mv
