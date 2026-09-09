#include "cs2mv/http_server.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include "net.h"

namespace cs2mv {
namespace {

const char* StatusText(int status) {
  switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 504: return "Gateway Timeout";
    default: return "Unknown";
  }
}

std::string ContentTypeFor(const std::string& path) {
  const std::size_t dot = path.rfind('.');
  const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
  if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
  if (ext == ".js") return "text/javascript; charset=utf-8";
  if (ext == ".css") return "text/css; charset=utf-8";
  if (ext == ".json") return "application/json; charset=utf-8";
  if (ext == ".svg") return "image/svg+xml";
  if (ext == ".png") return "image/png";
  if (ext == ".ico") return "image/x-icon";
  return "application/octet-stream";
}

// Rejects anything that could escape the static root: absolute paths, drive
// letters, and any ".." component.
bool IsSafeRelativePath(const std::string& path) {
  if (path.empty()) return false;
  if (path[0] == '/' || path[0] == '\\') return false;
  if (path.find(':') != std::string::npos) return false;
  std::size_t start = 0;
  while (start <= path.size()) {
    const std::size_t slash = path.find_first_of("/\\", start);
    const std::string part =
        path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
    if (part == "..") return false;
    if (slash == std::string::npos) break;
    start = slash + 1;
  }
  return true;
}

void ParseQuery(const std::string& raw, std::map<std::string, std::string>* out) {
  std::size_t pos = 0;
  while (pos < raw.size()) {
    const std::size_t amp = raw.find('&', pos);
    const std::string pair =
        raw.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
    const std::size_t eq = pair.find('=');
    if (eq != std::string::npos) {
      (*out)[HttpServer::UrlDecode(pair.substr(0, eq))] =
          HttpServer::UrlDecode(pair.substr(eq + 1));
    } else if (!pair.empty()) {
      (*out)[HttpServer::UrlDecode(pair)] = "";
    }
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
}

bool ParseRequest(const std::string& head, const std::string& body,
                  HttpRequest* out) {
  std::istringstream in(head);
  std::string line;
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();

  std::istringstream request_line(line);
  std::string target;
  if (!(request_line >> out->method >> target)) return false;

  const std::size_t question = target.find('?');
  if (question == std::string::npos) {
    out->path = HttpServer::UrlDecode(target);
  } else {
    out->path = HttpServer::UrlDecode(target.substr(0, question));
    out->raw_query = target.substr(question + 1);
    ParseQuery(out->raw_query, &out->query);
  }

  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) break;
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = line.substr(0, colon);
    for (char& c : name) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    std::size_t vstart = colon + 1;
    while (vstart < line.size() && line[vstart] == ' ') ++vstart;
    out->headers[name] = line.substr(vstart);
  }
  out->body = body;
  return true;
}

}  // namespace

void HttpResponse::SetError(int code, const std::string& message) {
  status = code;
  content_type = "application/json; charset=utf-8";
  std::string escaped;
  for (char c : message) {
    if (c == '"' || c == '\\') escaped += '\\';
    if (c == '\n' || c == '\r') {
      escaped += ' ';
      continue;
    }
    escaped += c;
  }
  body = "{\"error\":\"" + escaped + "\"}";
}

std::string HttpServer::UrlDecode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      out += ' ';
    } else if (s[i] == '%' && i + 2 < s.size()) {
      const std::string hex = s.substr(i + 1, 2);
      char* end = nullptr;
      const long v = std::strtol(hex.c_str(), &end, 16);
      if (end != nullptr && *end == '\0') {
        out += static_cast<char>(v);
        i += 2;
      } else {
        out += s[i];
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

void HttpServer::Route(const std::string& path, Handler handler) {
  routes_[path] = std::move(handler);
}

bool HttpServer::Start(const std::string& bind_address, int port,
                       std::string* error) {
  if (!net::EnsureInitialised()) {
    if (error != nullptr) *error = "socket startup failed";
    return false;
  }

  const net::Socket sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (sock == net::kInvalidSocket) {
    if (error != nullptr) *error = "cannot create socket";
    return false;
  }
  const int reuse = 1;
  ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  if (bind_address.empty() || bind_address == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, bind_address.c_str(), &addr.sin_addr) != 1) {
    net::CloseSocket(sock);
    if (error != nullptr) *error = "bad bind address: " + bind_address;
    return false;
  }

  if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    net::CloseSocket(sock);
    if (error != nullptr) {
      *error = "cannot bind " + bind_address + ":" + std::to_string(port) +
               " (already in use?)";
    }
    return false;
  }
  if (::listen(sock, 16) != 0) {
    net::CloseSocket(sock);
    if (error != nullptr) *error = "listen failed";
    return false;
  }

  sockaddr_in bound;
  std::memset(&bound, 0, sizeof(bound));
#ifdef _WIN32
  int bound_len = sizeof(bound);
#else
  socklen_t bound_len = sizeof(bound);
#endif
  if (::getsockname(sock, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0) {
    port_ = ntohs(bound.sin_port);
  } else {
    port_ = port;
  }

  listen_socket_ = static_cast<long long>(sock);
  running_ = true;
  return true;
}

void HttpServer::Stop() {
  running_ = false;
  if (listen_socket_ != -1) {
    net::CloseSocket(static_cast<net::Socket>(listen_socket_));
    listen_socket_ = -1;
  }
}

void HttpServer::Serve() {
  while (running_) {
    const net::Socket client = ::accept(
        static_cast<net::Socket>(listen_socket_), nullptr, nullptr);
    if (client == net::kInvalidSocket) {
      if (!running_) break;
      continue;
    }
    std::thread(&HttpServer::HandleConnection, this,
                static_cast<long long>(client))
        .detach();
  }
}

void HttpServer::HandleConnection(long long socket_handle) {
  const net::Socket sock = static_cast<net::Socket>(socket_handle);

  std::string buffer;
  char chunk[8192];
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    const int n = ::recv(sock, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      net::CloseSocket(sock);
      return;
    }
    buffer.append(chunk, static_cast<std::size_t>(n));
    header_end = buffer.find("\r\n\r\n");
    if (buffer.size() > (1u << 20)) {
      net::CloseSocket(sock);
      return;
    }
  }

  const std::string head = buffer.substr(0, header_end + 2);
  std::string body = buffer.substr(header_end + 4);

  HttpRequest request;
  HttpResponse response;
  if (!ParseRequest(head, body, &request)) {
    response.SetError(400, "malformed request");
  } else {
    // Read the rest of the body when one was announced.
    auto it = request.headers.find("content-length");
    if (it != request.headers.end()) {
      const std::size_t want = static_cast<std::size_t>(std::strtoull(
          it->second.c_str(), nullptr, 10));
      while (request.body.size() < want && want < (16u << 20)) {
        const int n = ::recv(sock, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        request.body.append(chunk, static_cast<std::size_t>(n));
      }
    }
    BuildResponse(request, &response);
  }

  std::ostringstream out;
  out << "HTTP/1.1 " << response.status << " " << StatusText(response.status)
      << "\r\n"
      << "Content-Type: " << response.content_type << "\r\n"
      << "Content-Length: " << response.body.size() << "\r\n"
      << "Cache-Control: no-store\r\n"
      << "Connection: close\r\n";
  for (const auto& header : response.extra_headers) {
    out << header.first << ": " << header.second << "\r\n";
  }
  out << "\r\n";

  const std::string head_text = out.str();
  net::SendAll(sock, head_text);
  if (request.method != "HEAD") net::SendAll(sock, response.body);
  net::CloseSocket(sock);
}

bool HttpServer::BuildResponse(const HttpRequest& request,
                               HttpResponse* response) {
  auto route = routes_.find(request.path);
  if (route != routes_.end()) {
    route->second(request, response);
    return true;
  }
  if (!static_root_.empty()) {
    std::string relative = request.path == "/" ? "index.html" : request.path;
    if (!relative.empty() && relative[0] == '/') relative.erase(0, 1);
    if (!IsSafeRelativePath(relative)) {
      response->SetError(403, "forbidden path");
      return true;
    }
    if (ServeFile(static_root_ + "/" + relative, response)) return true;
  }
  response->SetError(404, "no route for " + request.path);
  return true;
}

bool HttpServer::ServeFile(const std::string& path, HttpResponse* response) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  response->status = 200;
  response->content_type = ContentTypeFor(path);
  response->body = ss.str();
  return true;
}

}  // namespace cs2mv
