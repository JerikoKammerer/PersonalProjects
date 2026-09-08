#include "cs2mv/locator.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "cs2mv/bzip2.h"
#include "cs2mv/http_client.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace cs2mv {
namespace {

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

bool MakeDirectory(const std::string& path) {
#ifdef _WIN32
  return ::_mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
  return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

std::string Trim(const std::string& s) {
  std::size_t begin = 0;
  std::size_t end = s.size();
  while (begin < end && (s[begin] == ' ' || s[begin] == '\t' || s[begin] == '\r')) ++begin;
  while (end > begin && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r')) --end;
  return s.substr(begin, end - begin);
}

bool LooksLikeUrl(const std::string& s) {
  return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

}  // namespace

bool DemoIndex::Load(const std::string& path, std::string* error) {
  std::ifstream f(path);
  if (!f) return true;  // no index yet
  std::string line;
  int line_number = 0;
  while (std::getline(f, line)) {
    ++line_number;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;
    const std::size_t sep = trimmed.find_first_of("\t ");
    if (sep == std::string::npos) {
      return Err(error, path + ":" + std::to_string(line_number) +
                            ": expected '<share code or match id> <path or URL>'");
    }
    entries_[Trim(trimmed.substr(0, sep))] = Trim(trimmed.substr(sep + 1));
  }
  return true;
}

bool DemoIndex::Save(const std::string& path, std::string* error) const {
  std::ofstream f(path, std::ios::trunc);
  if (!f) return Err(error, "cannot write " + path);
  f << "# cs2-match-viewer demo index\n"
    << "# <share code or match id><tab><demo path or http URL>\n";
  for (const auto& entry : entries_) {
    f << entry.first << "\t" << entry.second << "\n";
  }
  return static_cast<bool>(f);
}

void DemoIndex::Set(const std::string& key, const std::string& location) {
  entries_[key] = location;
}

bool DemoIndex::Lookup(const std::string& key, DemoLocation* out) const {
  auto it = entries_.find(key);
  if (it == entries_.end()) return false;
  out->kind = LooksLikeUrl(it->second) ? DemoLocation::kUrl : DemoLocation::kFile;
  out->value = it->second;
  return true;
}

namespace {

// Where Steam itself is installed. The registry is what makes this reliable:
// a library can live anywhere, and on this machine Steam sits under a OneDrive
// folder that no hardcoded guess would find.
std::vector<std::string> SteamRoots() {
  std::vector<std::string> roots;
#ifdef _WIN32
  char buffer[1024];
  DWORD size = sizeof(buffer);
  if (::RegGetValueA(HKEY_CURRENT_USER, "Software\\Valve\\Steam", "SteamPath",
                     RRF_RT_REG_SZ, nullptr, buffer, &size) == ERROR_SUCCESS) {
    roots.push_back(buffer);
  }
  size = sizeof(buffer);
  if (::RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\Valve\\Steam",
                     "InstallPath", RRF_RT_REG_SZ, nullptr, buffer,
                     &size) == ERROR_SUCCESS) {
    roots.push_back(buffer);
  }
  const char* programs = std::getenv("ProgramFiles(x86)");
  if (programs != nullptr) roots.push_back(std::string(programs) + "\\Steam");
#else
  const char* home = std::getenv("HOME");
  if (home != nullptr) {
    const std::string h(home);
    roots.push_back(h + "/.steam/steam");
    roots.push_back(h + "/.local/share/Steam");
    roots.push_back(h + "/Library/Application Support/Steam");
  }
#endif
  return roots;
}

// steamapps/libraryfolders.vdf lists every library folder, in Valve's
// KeyValues format. Only the "path" entries are of interest:
//
//   "path"    "D:\\SteamLibrary"
std::vector<std::string> LibrariesIn(const std::string& steam_root) {
  std::vector<std::string> libraries;
  libraries.push_back(steam_root);

  std::ifstream f(steam_root + "/steamapps/libraryfolders.vdf");
  if (!f) return libraries;
  std::string line;
  while (std::getline(f, line)) {
    const std::size_t key = line.find("\"path\"");
    if (key == std::string::npos) continue;
    const std::size_t open = line.find('"', key + 6);
    if (open == std::string::npos) continue;
    const std::size_t close = line.find('"', open + 1);
    if (close == std::string::npos) continue;

    std::string path = line.substr(open + 1, close - open - 1);
    // KeyValues escapes backslashes.
    std::string unescaped;
    for (std::size_t i = 0; i < path.size(); ++i) {
      unescaped.push_back(path[i]);
      if (path[i] == '\\' && i + 1 < path.size() && path[i + 1] == '\\') ++i;
    }
    if (!unescaped.empty()) libraries.push_back(unescaped);
  }
  return libraries;
}

}  // namespace

std::vector<std::string> Cs2ReplayDirectories() {
  std::vector<std::string> found;
  for (const std::string& root : SteamRoots()) {
    for (const std::string& library : LibrariesIn(root)) {
      const std::string dir = library +
          "/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays";
      std::error_code ec;
      if (!std::filesystem::is_directory(dir, ec)) continue;
      // Different roots routinely resolve to the same directory.
      const std::string key = std::filesystem::absolute(dir, ec).string();
      bool seen = false;
      for (const std::string& existing : found) {
        std::error_code cmp;
        if (std::filesystem::equivalent(existing, key, cmp)) {
          seen = true;
          break;
        }
      }
      if (!seen) found.push_back(key);
    }
  }
  return found;
}

namespace {

// Pulls the id out of a CS2 demo filename:
//
//   match730_003841497038206271665_1184512206_392.dem
//            ^^^^^^^^^^^^^^^^^^^^^ zero padded reservation id
//
// Parsed out structurally rather than searched for: a substring test would
// match any filename that merely contains the digits, which for a small id is
// very nearly all of them. Returns 0 when the name is not of this shape.
std::uint64_t DemoFileId(const std::string& filename) {
  static const char kPrefix[] = "match730_";
  const std::size_t prefix_len = sizeof(kPrefix) - 1;
  if (filename.compare(0, prefix_len, kPrefix) != 0) return 0;

  const std::size_t end = filename.find('_', prefix_len);
  if (end == std::string::npos) return 0;
  const std::string digits = filename.substr(prefix_len, end - prefix_len);
  if (digits.empty() || digits.size() > 21) return 0;

  std::uint64_t value = 0;
  for (char c : digits) {
    if (c < '0' || c > '9') return 0;
    const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return 0;
    value = value * 10 + digit;
  }
  return value;
}

}  // namespace

bool FindDownloadedDemo(const ShareCode& code, std::string* path) {
  if (code.outcome_id == 0 && code.match_id == 0) return false;

  for (const std::string& dir : Cs2ReplayDirectories()) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    if (ec) continue;
    std::string by_match;
    for (; it != end; it.increment(ec)) {
      if (ec) break;
      const std::filesystem::path& p = it->path();
      if (p.extension() != ".dem") continue;

      const std::uint64_t id = DemoFileId(p.filename().string());
      if (id == 0) continue;
      // CS2 names the file after the reservation id, which is the share code's
      // outcome id. The match id is checked too, in case that ever changes.
      if (id == code.outcome_id) {
        *path = p.string();
        return true;
      }
      if (by_match.empty() && id == code.match_id) by_match = p.string();
    }
    if (!by_match.empty()) {
      *path = by_match;
      return true;
    }
  }
  return false;
}

std::string DefaultCacheDir() {
#ifdef _WIN32
  const char* base = std::getenv("LOCALAPPDATA");
  if (base == nullptr) base = std::getenv("TEMP");
  const std::string root = base != nullptr ? base : ".";
  return root + "\\cs2-match-viewer";
#else
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg != nullptr && xdg[0] != '\0') return std::string(xdg) + "/cs2-match-viewer";
  const char* home = std::getenv("HOME");
  const std::string root = home != nullptr ? std::string(home) + "/.cache" : ".";
  return root + "/cs2-match-viewer";
#endif
}

bool FetchDemo(const std::string& url, const std::string& dest,
               const ResolveOptions& options, std::string* error) {
  HttpGetOptions get;
  if (options.progress) {
    auto callback = options.progress;
    get.progress = [callback](std::uint64_t done, std::uint64_t total) {
      callback(done, total);
      return true;
    };
  }

  HttpGetResult result;
  if (!HttpGet(url, get, &result, error)) return false;
  if (result.status != 200) {
    return Err(error, "download failed: HTTP " + std::to_string(result.status) +
                          " from " + url +
                          (result.status == 404
                               ? " (Valve keeps match demos for about 30 days)"
                               : ""));
  }

  std::string payload;
  if (IsBzip2(result.body.data(), result.body.size())) {
    std::string bz_error;
    if (!Bzip2Uncompress(result.body.data(), result.body.size(), &payload,
                         &bz_error)) {
      return Err(error, "cannot unpack downloaded demo: " + bz_error);
    }
  } else {
    payload.swap(result.body);
  }

  std::ofstream out(dest, std::ios::binary | std::ios::trunc);
  if (!out) return Err(error, "cannot write " + dest);
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!out) return Err(error, "failed while writing " + dest);
  return true;
}

bool ResolveDemo(const ShareCode& code, const std::string& code_text,
                 const ResolveOptions& options, std::string* demo_path,
                 std::string* error) {
  std::string cache_dir = options.cache_dir.empty() ? DefaultCacheDir() : options.cache_dir;
  MakeDirectory(cache_dir);
  const std::string index_path =
      options.index_path.empty() ? cache_dir + "/index.txt" : options.index_path;

  std::string canonical = code_text;
  NormalizeShareCode(code_text, &canonical, nullptr);
  const std::string match_id_text = std::to_string(code.match_id);
  const std::string cached = cache_dir + "/" + match_id_text + ".dem";

  DemoIndex index;
  std::string index_error;
  if (!index.Load(index_path, &index_error)) return Err(error, index_error);

  DemoLocation location;
  const bool indexed =
      index.Lookup(canonical, &location) || index.Lookup(match_id_text, &location);

  // An explicit file entry is the user's own choice, so it wins outright.
  if (indexed && location.kind == DemoLocation::kFile) {
    if (!FileExists(location.value)) {
      return Err(error, "the index points at " + location.value +
                            " but that file is missing");
    }
    *demo_path = location.value;
    return true;
  }

  if (FileExists(cached)) {
    *demo_path = cached;
    return true;
  }
  const std::string compressed = cache_dir + "/" + match_id_text + ".dem.bz2";
  if (FileExists(compressed)) {
    *demo_path = compressed;
    return true;
  }

  // A demo the game already downloaded beats fetching one over the network,
  // and needs no URL at all. Checked before any indexed URL so that a stale or
  // wrong link cannot mask a perfectly good local file.
  std::string downloaded;
  if (FindDownloadedDemo(code, &downloaded)) {
    *demo_path = downloaded;
    // Remember it, so the next lookup skips the search and `cs2mv list` shows
    // where the match came from.
    index.Set(canonical, downloaded);
    index.Save(index_path, nullptr);
    return true;
  }

  if (indexed && location.kind == DemoLocation::kUrl) {
    if (!options.allow_download) {
      return Err(error, "downloads are disabled; the index has a URL for this match");
    }
    if (!FetchDemo(location.value, cached, options, error)) return false;
    *demo_path = cached;
    return true;
  }

  const std::vector<std::string> replay_dirs = Cs2ReplayDirectories();
  std::string message =
      "That match is not on this machine yet.\n"
      "\n"
      "Easiest fix: open CS2, go to Watch -> Your Matches, find this match and "
      "click Download. Then paste the share code again - it will be picked up "
      "automatically, with nothing else to do.\n"
      "\n"
      "A share code carries no download link of its own (just match id " +
      match_id_text +
      ", an outcome id and a token), and only the CS2 game coordinator can turn "
      "one into a URL, which needs a logged-in Steam session.\n";
  if (replay_dirs.empty()) {
    message += "\nNo CS2 replay folder was found on this machine, so the "
               "automatic lookup had nowhere to look.";
  } else {
    message += "\nLooked in:\n";
    for (const std::string& dir : replay_dirs) message += "  " + dir + "\n";
  }
  message += "\nOr point at it directly, if you have the file or a URL:\n"
             "  cs2mv add " + canonical + " C:\\path\\to\\match.dem\n"
             "  cs2mv add " + canonical + " http://replay123.valve.net/730/...dem.bz2\n"
             "(the index lives at " + index_path + ")";
  return Err(error, message);
}

}  // namespace cs2mv

