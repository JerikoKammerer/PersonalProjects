// A minimal HTTP/1.1 GET client, used to pull demos off Valve's replay
// servers.
//
// Plain HTTP only, deliberately: replay URLs handed out by the game
// coordinator are http://replayNNN.valve.net/... and adding TLS would mean
// taking on a crypto dependency for no gain. An https:// URL is rejected with
// an explanatory error rather than silently downgraded.
#ifndef CS2MV_HTTP_CLIENT_H_
#define CS2MV_HTTP_CLIENT_H_

#include <cstdint>
#include <functional>
#include <string>

namespace cs2mv {

struct HttpGetResult {
  int status = 0;
  std::string body;
  std::string content_type;
};

// Called with (bytes so far, total or 0 when unknown). Return false to abort.
using ProgressCallback = std::function<bool(std::uint64_t, std::uint64_t)>;

struct HttpGetOptions {
  int timeout_seconds = 60;
  int max_redirects = 5;
  // Guards against a hostile or wrong URL filling memory. Demos run to a few
  // hundred MB, so the default is generous.
  std::uint64_t max_body_bytes = 2ull * 1024 * 1024 * 1024;
  ProgressCallback progress;
};

// Performs a GET, following redirects. Returns false on transport failure;
// an HTTP error status is reported through `out->status` with a true return.
bool HttpGet(const std::string& url, const HttpGetOptions& options,
             HttpGetResult* out, std::string* error);

// Splits a URL into parts. Returns false when it is not an absolute http URL.
bool ParseHttpUrl(const std::string& url, std::string* host, int* port,
                  std::string* path, std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_HTTP_CLIENT_H_
