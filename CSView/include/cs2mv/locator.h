// Getting from a share code to a demo file on disk.
//
// A share code carries no URL. The only way to turn one into a download link
// is to ask the CS2 game coordinator, and the GC only talks to a logged-in
// Steam client, which means a Steam session, an app ticket and the GC message
// framing - a dependency this project does not take on. See README.md,
// "Getting the demo".
//
// So resolution goes through a small index the user controls:
//
//   1. an entry in the index file, keyed by share code or match id, holding
//      either a local path or an http URL,
//   2. a file already sitting in the cache directory, named after the match id,
//   3. otherwise a failure that says exactly what to add and how.
//
// A URL entry is downloaded and unpacked into the cache on first use.
#ifndef CS2MV_LOCATOR_H_
#define CS2MV_LOCATOR_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "cs2mv/sharecode.h"

namespace cs2mv {

struct DemoLocation {
  enum Kind { kNone, kFile, kUrl };
  Kind kind = kNone;
  std::string value;
};

// A text file of "key<tab>location" lines. Keys are share codes in canonical
// form, or decimal match ids. Lines starting with '#' are comments.
class DemoIndex {
 public:
  // A missing file is not an error: it just means an empty index.
  bool Load(const std::string& path, std::string* error);
  bool Save(const std::string& path, std::string* error) const;

  void Set(const std::string& key, const std::string& location);
  bool Lookup(const std::string& key, DemoLocation* out) const;

  const std::map<std::string, std::string>& entries() const { return entries_; }

 private:
  std::map<std::string, std::string> entries_;
};

struct ResolveOptions {
  std::string index_path;   // defaults to <cache_dir>/index.txt
  std::string cache_dir;    // where downloads are unpacked
  bool allow_download = true;
  // Reports download progress as (bytes, total). Total is 0 when unknown.
  std::function<void(std::uint64_t, std::uint64_t)> progress;
};

// Fills `*demo_path` with a readable .dem for `code`. Returns false with a
// user-facing explanation in `*error` when the demo cannot be found.
bool ResolveDemo(const ShareCode& code, const std::string& code_text,
                 const ResolveOptions& options, std::string* demo_path,
                 std::string* error);

// Downloads `url`, unpacks it if it is bzip2, and writes it to `dest`.
bool FetchDemo(const std::string& url, const std::string& dest,
               const ResolveOptions& options, std::string* error);

// Default cache directory: %LOCALAPPDATA%/cs2-match-viewer on Windows,
// $XDG_CACHE_HOME or ~/.cache/cs2-match-viewer elsewhere.
std::string DefaultCacheDir();

// Every CS2 replay directory on this machine, across all Steam library
// folders. Empty when Steam or CS2 is not installed.
std::vector<std::string> Cs2ReplayDirectories();

// Looks for a demo the game already downloaded, so that clicking Download in
// CS2 is the whole workflow and no URL is needed.
//
// CS2 names them match730_<21 digit id>_<...>.dem, and that id is the share
// code's *outcome* id (the reservation id), not the match id - both are
// checked. Returns false when nothing matches.
bool FindDownloadedDemo(const ShareCode& code, std::string* path);

}  // namespace cs2mv

#endif  // CS2MV_LOCATOR_H_
