#include "cs2mv/locator.h"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

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
  if (index.Lookup(canonical, &location) || index.Lookup(match_id_text, &location)) {
    if (location.kind == DemoLocation::kFile) {
      if (!FileExists(location.value)) {
        return Err(error, "the index points at " + location.value +
                              " but that file is missing");
      }
      *demo_path = location.value;
      return true;
    }
    if (!options.allow_download) {
      return Err(error, "downloads are disabled; the index has a URL for this match");
    }
    if (!FileExists(cached) && !FetchDemo(location.value, cached, options, error)) {
      return false;
    }
    *demo_path = cached;
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

  return Err(error,
             "no demo on hand for match " + match_id_text + ".\n"
             "A share code does not contain a download URL - only the CS2 game "
             "coordinator can turn one into a URL, and that needs a logged-in "
             "Steam session.\n"
             "Once you have the URL or the file, register it:\n"
             "  cs2mv add " + canonical + " http://replayNNN.valve.net/730/....dem.bz2\n"
             "  cs2mv add " + canonical + " C:\\path\\to\\match.dem\n"
             "(the index lives at " + index_path + ")");
}

}  // namespace cs2mv
