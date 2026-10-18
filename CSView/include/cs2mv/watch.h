// Handing a demo to CS2 for playback.
//
// A demo file contains no video. It is a tick-by-tick record of network state,
// so watching one means something has to *render* it, and the only thing that
// can render it faithfully is CS2 itself. This hands the file to the game.
//
// The alternative - drawing the match in the browser - needs entity state
// (player positions every tick), which is the one part of a Source 2 demo this
// project deliberately does not decode. See README.md.
#ifndef CS2MV_WATCH_H_
#define CS2MV_WATCH_H_

#include <string>

namespace cs2mv {

// How CS2 should be told to refer to a demo.
//
// `playdemo` resolves relative to the game's csgo directory, which is why a
// demo sitting in replays/ can be named without quoting a path full of spaces.
// Returns false when the file is not under any CS2 replay folder.
bool Cs2DemoReference(const std::string& demo_path, std::string* reference);

// Copies a demo into CS2's replay folder so that it can be played. Needed for
// demos fetched by the game coordinator, which land in the cache rather than
// where CS2 looks. `dest` receives the new path.
bool CopyIntoReplays(const std::string& demo_path, std::string* dest,
                     std::string* error);

// Launches CS2 playing `demo_path`.
//
// `dry_run` reports what would happen and changes nothing at all - it neither
// starts the game nor copies any file - which is how this gets tested without
// disturbing somebody's machine. `command` receives what was (or would be)
// run, and `requires_copy` says whether the demo has to be copied into CS2's
// replay folder first, which for a 200 MB file is worth knowing in advance.
bool LaunchCs2Playback(const std::string& demo_path, bool dry_run,
                       std::string* command, bool* requires_copy,
                       std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_WATCH_H_
