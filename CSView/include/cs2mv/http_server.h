// A small blocking HTTP/1.1 server for the local web UI.
//
// Scope: it binds to the loopback interface by default, speaks HTTP/1.0 style
// "one request per connection", and serves a handful of routes plus a static
// directory. It is a viewer front end for a desktop tool, not a public web
// server, and it should not be exposed to an untrusted network.
#ifndef CS2MV_HTTP_SERVER_H_
#define CS2MV_HTTP_SERVER_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cs2mv {

struct HttpRequest {
  std::string method;
  std::string path;                            // decoded, without the query
  std::string raw_query;
  std::map<std::string, std::string> query;    // decoded key/value pairs
  std::map<std::string, std::string> headers;  // lowercase names
  std::string body;

  std::string Param(const std::string& key) const {
    auto it = query.find(key);
    return it == query.end() ? std::string() : it->second;
  }
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "text/html; charset=utf-8";
  std::string body;
  std::vector<std::pair<std::string, std::string>> extra_headers;

  void SetJson(std::string json) {
    content_type = "application/json; charset=utf-8";
    body = std::move(json);
  }
  void SetError(int code, const std::string& message);
};

class HttpServer {
 public:
  using Handler = std::function<void(const HttpRequest&, HttpResponse*)>;

  // Binds and listens. `port` may be 0 to let the OS choose, in which case
  // port() reports what was picked.
  bool Start(const std::string& bind_address, int port, std::string* error);

  // Handles connections until Stop() is called. One thread per connection.
  void Serve();
  void Stop();

  // Registers an exact-path route. Later registrations win.
  void Route(const std::string& path, Handler handler);

  // Serves files below `directory` for any path not matched by a route.
  // Directory traversal outside the root is refused.
  void ServeStatic(const std::string& directory) { static_root_ = directory; }

  int port() const { return port_; }

  // Percent-decoding, exposed for tests.
  static std::string UrlDecode(const std::string& s);

 private:
  void HandleConnection(long long socket_handle);
  bool BuildResponse(const HttpRequest& request, HttpResponse* response);
  bool ServeFile(const std::string& path, HttpResponse* response);

  std::map<std::string, Handler> routes_;
  std::string static_root_;
  long long listen_socket_ = -1;
  int port_ = 0;
  bool running_ = false;
};

}  // namespace cs2mv

#endif  // CS2MV_HTTP_SERVER_H_
