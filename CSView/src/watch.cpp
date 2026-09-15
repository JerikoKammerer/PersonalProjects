#include "cs2mv/watch.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <vector>

#include "cs2mv/locator.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace cs2mv {
namespace {

constexpr int kCs2AppId = 730;

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

}  // namespace

bool Cs2DemoReference(const std::string& demo_path, std::string* reference) {
  std::error_code ec;
  const std::filesystem::path demo = std::filesystem::absolute(demo_path, ec);
  if (ec) return false;

  for (const std::string& dir : Cs2ReplayDirectories()) {
    const std::filesystem::path replays = std::filesystem::absolute(dir, ec);
    if (ec) continue;
    if (!std::filesystem::equivalent(demo.parent_path(), replays, ec)) continue;
    // CS2 looks under game/csgo, and replays/ sits directly inside it.
    *reference = "replays/" + demo.filename().string();
    return true;
  }
  return false;
}

bool CopyIntoReplays(const std::string& demo_path, std::string* dest,
                     std::string* error) {
  const std::vector<std::string> dirs = Cs2ReplayDirectories();
  if (dirs.empty()) {
    return Err(error, "no CS2 replay folder found, so there is nowhere to put "
                      "the demo for playback");
  }

  std::error_code ec;
  const std::filesystem::path source = std::filesystem::absolute(demo_path, ec);
  const std::filesystem::path target =
      std::filesystem::path(dirs.front()) / source.filename();

  if (std::filesystem::exists(target, ec)) {
    *dest = target.string();
    return true;
  }
  std::filesystem::copy_file(source, target,
                             std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    return Err(error, "could not copy the demo into CS2's replay folder: " +
                          ec.message());
  }
  *dest = target.string();
  return true;
}

bool LaunchCs2Playback(const std::string& demo_path, bool dry_run,
                       std::string* command, bool* requires_copy,
                       std::string* error) {
  *requires_copy = false;
  std::string reference;
  if (!Cs2DemoReference(demo_path, &reference)) {
    *requires_copy = true;
    if (dry_run) {
      // A dry run must not touch anything, so work out the name the copy would
      // take rather than making it.
      std::error_code ec;
      const std::filesystem::path source =
          std::filesystem::absolute(demo_path, ec);
      *command = "steam://run/" + std::to_string(kCs2AppId) + "//+playdemo replays/" +
                 source.filename().string();
      return true;
    }
    std::string copied;
    if (!CopyIntoReplays(demo_path, &copied, error)) return false;
    if (!Cs2DemoReference(copied, &reference)) {
      return Err(error, "copied the demo but CS2 still cannot see it");
    }
  }

  // steam://run hands the launch to Steam, which knows where CS2 is and starts
  // it if it is not already running. The reference has no spaces, so nothing
  // here needs quoting or escaping.
  *command = "steam://run/" + std::to_string(kCs2AppId) + "//+playdemo " + reference;
  if (dry_run) return true;

#ifdef _WIN32
  const HINSTANCE result = ::ShellExecuteA(nullptr, "open", command->c_str(),
                                           nullptr, nullptr, SW_SHOWNORMAL);
  // ShellExecute returns a value <= 32 to report failure.
  if (reinterpret_cast<INT_PTR>(result) <= 32) {
    return Err(error, "could not hand the demo to Steam (is Steam installed?)");
  }
  return true;
#elif defined(__APPLE__)
  const std::string line = "open \"" + *command + "\" >/dev/null 2>&1 &";
  return std::system(line.c_str()) == 0
             ? true
             : Err(error, "could not run `open` for the steam:// link");
#else
  const std::string line = "xdg-open \"" + *command + "\" >/dev/null 2>&1 &";
  return std::system(line.c_str()) == 0
             ? true
             : Err(error, "could not run `xdg-open` for the steam:// link");
#endif
}

}  // namespace cs2mv
