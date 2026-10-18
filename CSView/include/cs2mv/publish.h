// Writing a match out as plain files, so that a web server which can only
// serve files - GitHub Pages, say - can still show it.
//
// The viewer normally answers /api/match and /api/replay by parsing a demo on
// demand, which needs the demo, the parser, and a few hundred megabytes of
// memory. None of that exists on a static host. Publishing does the same work
// once, here, and writes the answers to disk under the same names the page
// asks for:
//
//   <out>/index.json          every published match, newest first
//   <out>/<id>/match.json     what /api/match returns
//   <out>/<id>/round-<n>.json what /api/replay?round=<n> returns
//   <out>/<id>/map.png        what /api/replay/map returns
//   <out>/<id>/walked.png     the same with ?raw=1
//
// `id` is the share code when there is one, so a published match keeps a
// stable URL.
#ifndef CS2MV_PUBLISH_H_
#define CS2MV_PUBLISH_H_

#include <string>

#include "cs2mv/match.h"
#include "cs2mv/replay.h"

namespace cs2mv {

// Writes one match's files under `out_dir`, creating directories as needed.
// `demo_path` names the demo it came from, whose modification time dates the
// match in the catalogue. The id used is returned in `id`.
bool PublishMatch(const Match& match, const Replay& replay,
                  const std::string& out_dir, const std::string& demo_path,
                  std::string* id, std::string* error);

// Rebuilds <out_dir>/index.json from the card every published match leaves
// behind, so that publishing one match does not disturb the others.
bool WriteCatalog(const std::string& out_dir, std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_PUBLISH_H_
