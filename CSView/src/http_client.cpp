#include "cs2mv/http_client.h"

#include <cstdlib>
#include <cstring>

#include "net.h"

namespace cs2mv {
namespace {

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

std::string ToLower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// Reads until `needle` is present or the connection ends. Everything read is
// appended to `*buffer`; the caller keeps the tail after the needle.
bool ReadUntil(net::Socket sock, const std::string& needle, std::string* buffer) {
  char chunk[8192];
  while (buffer->find(needle) == std::string::npos) {
    const int n = ::recv(sock, chunk, sizeof(chunk), 0);
    if (n <= 0) return false;
    buffer->append(chunk, static_cast<std::size_t>(n));
    if (buffer->size() > (1u << 20)) return false;  // header block gone wild
  }
  return true;
}

struct Headers {
  int status = 0;
  std::string location;
  std::string content_type;
  std::uint64_t content_length = 0;
  bool has_content_length = false;
  bool chunked = false;
};

bool ParseHeaders(const std::string& block, Headers* out, std::string* error) {
  std::size_t line_end = block.find("\r\n");
  if (line_end == std::string::npos) return Err(error, "malformed HTTP response");
  const std::string status_line = block.substr(0, line_end);
  const std::size_t sp = status_line.find(' ');
  if (sp == std::string::npos) return Err(error, "malformed HTTP status line");
  out->status = std::atoi(status_line.c_str() + sp + 1);

  std::size_t pos = line_end + 2;
  while (pos < block.size()) {
    const std::size_t end = block.find("\r\n", pos);
    if (end == std::string::npos || end == pos) break;
    const std::string line = block.substr(pos, end - pos);
    pos = end + 2;
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string name = ToLower(line.substr(0, colon));
    std::size_t vstart = colon + 1;
    while (vstart < line.size() && line[vstart] == ' ') ++vstart;
    const std::string value = line.substr(vstart);

    if (name == "location") {
      out->location = value;
    } else if (name == "content-type") {
      out->content_type = value;
    } else if (name == "content-length") {
      out->content_length = std::strtoull(value.c_str(), nullptr, 10);
      out->has_content_length = true;
    } else if (name == "transfer-encoding") {
      out->chunked = ToLower(value).find("chunked") != std::string::npos;
    }
  }
  return true;
}

bool Connect(const std::string& host, int port, int timeout_seconds,
             net::Socket* out, std::string* error) {
  addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  const std::string port_str = std::to_string(port);
  addrinfo* result = nullptr;
  if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0 ||
      result == nullptr) {
    return Err(error, "cannot resolve host " + host);
  }

  net::Socket sock = net::kInvalidSocket;
  for (addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
    sock = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock == net::kInvalidSocket) continue;
    if (::connect(sock, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) break;
    net::CloseSocket(sock);
    sock = net::kInvalidSocket;
  }
  ::freeaddrinfo(result);
  if (sock == net::kInvalidSocket) {
    return Err(error, "cannot connect to " + host + ":" + port_str);
  }

#ifdef _WIN32
  const DWORD timeout_ms = static_cast<DWORD>(timeout_seconds) * 1000;
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
  ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
  timeval tv;
  tv.tv_sec = timeout_seconds;
  tv.tv_usec = 0;
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
  *out = sock;
  return true;
}

// Consumes a chunked body from `sock`, seeding with whatever is already in
// `pending`.
bool ReadChunked(net::Socket sock, std::string pending, std::string* body,
                 const HttpGetOptions& options, std::string* error) {
  char chunk[16384];
  for (;;) {
    std::size_t line_end = pending.find("\r\n");
    while (line_end == std::string::npos) {
      const int n = ::recv(sock, chunk, sizeof(chunk), 0);
      if (n <= 0) return Err(error, "connection closed mid chunk header");
      pending.append(chunk, static_cast<std::size_t>(n));
      line_end = pending.find("\r\n");
    }
    const std::size_t size = std::strtoull(pending.c_str(), nullptr, 16);
    pending.erase(0, line_end + 2);
    if (size == 0) return true;
    if (body->size() + size > options.max_body_bytes) {
      return Err(error, "response exceeds the configured size limit");
    }
    while (pending.size() < size + 2) {
      const int n = ::recv(sock, chunk, sizeof(chunk), 0);
      if (n <= 0) return Err(error, "connection closed mid chunk body");
      pending.append(chunk, static_cast<std::size_t>(n));
      if (options.progress &&
          !options.progress(body->size() + pending.size(), 0)) {
        return Err(error, "download cancelled");
      }
    }
    body->append(pending, 0, size);
    pending.erase(0, size + 2);
  }
}

bool GetOnce(const std::string& url, const HttpGetOptions& options,
             HttpGetResult* out, std::string* redirect, std::string* error) {
  std::string host, path;
  int port = 80;
  if (!ParseHttpUrl(url, &host, &port, &path, error)) return false;

  net::Socket sock = net::kInvalidSocket;
  if (!Connect(host, port, options.timeout_seconds, &sock, error)) return false;

  const std::string request =
      "GET " + path + " HTTP/1.1\r\n"
      "Host: " + host + "\r\n"
      "User-Agent: cs2-match-viewer/1.0\r\n"
      "Accept: */*\r\n"
      "Connection: close\r\n\r\n";
  if (!net::SendAll(sock, request)) {
    net::CloseSocket(sock);
    return Err(error, "failed to send request to " + host);
  }

  std::string buffer;
  if (!ReadUntil(sock, "\r\n\r\n", &buffer)) {
    net::CloseSocket(sock);
    return Err(error, "no HTTP response from " + host);
  }
  const std::size_t header_end = buffer.find("\r\n\r\n");
  Headers headers;
  if (!ParseHeaders(buffer.substr(0, header_end + 2), &headers, error)) {
    net::CloseSocket(sock);
    return false;
  }
  std::string pending = buffer.substr(header_end + 4);

  out->status = headers.status;
  out->content_type = headers.content_type;

  if (headers.status >= 300 && headers.status < 400 && !headers.location.empty()) {
    *redirect = headers.location;
    net::CloseSocket(sock);
    return true;
  }

  bool ok = true;
  if (headers.chunked) {
    ok = ReadChunked(sock, pending, &out->body, options, error);
  } else {
    out->body = pending;
    const std::uint64_t total = headers.has_content_length ? headers.content_length : 0;
    char chunk[65536];
    for (;;) {
      if (headers.has_content_length && out->body.size() >= headers.content_length) break;
      const int n = ::recv(sock, chunk, sizeof(chunk), 0);
      if (n <= 0) break;  // a close ends the body when there is no length
      if (out->body.size() + static_cast<std::size_t>(n) > options.max_body_bytes) {
        ok = Err(error, "response exceeds the configured size limit");
        break;
      }
      out->body.append(chunk, static_cast<std::size_t>(n));
      if (options.progress && !options.progress(out->body.size(), total)) {
        ok = Err(error, "download cancelled");
        break;
      }
    }
    if (ok && headers.has_content_length && out->body.size() < headers.content_length) {
      ok = Err(error, "connection closed before the whole body arrived");
    }
  }

  net::CloseSocket(sock);
  return ok;
}

}  // namespace

bool ParseHttpUrl(const std::string& url, std::string* host, int* port,
                  std::string* path, std::string* error) {
  const std::string lower = ToLower(url);
  if (lower.rfind("https://", 0) == 0) {
    return Err(error,
               "https is not supported: this client speaks plain HTTP only. "
               "Valve replay URLs are http://; for anything else, download the "
               "demo yourself and pass the file.");
  }
  if (lower.rfind("http://", 0) != 0) {
    return Err(error, "not an absolute http:// URL: " + url);
  }

  const std::string rest = url.substr(7);
  const std::size_t slash = rest.find('/');
  std::string authority = slash == std::string::npos ? rest : rest.substr(0, slash);
  *path = slash == std::string::npos ? "/" : rest.substr(slash);
  if (authority.empty()) return Err(error, "URL has no host: " + url);

  const std::size_t colon = authority.rfind(':');
  *port = 80;
  if (colon != std::string::npos && authority.find(']') == std::string::npos) {
    *port = std::atoi(authority.c_str() + colon + 1);
    if (*port <= 0 || *port > 65535) return Err(error, "bad port in URL: " + url);
    authority = authority.substr(0, colon);
  }
  *host = authority;
  return true;
}

bool HttpGet(const std::string& url, const HttpGetOptions& options,
             HttpGetResult* out, std::string* error) {
  if (!net::EnsureInitialised()) return Err(error, "socket startup failed");

  std::string current = url;
  for (int hop = 0; hop <= options.max_redirects; ++hop) {
    HttpGetResult attempt;
    std::string redirect;
    if (!GetOnce(current, options, &attempt, &redirect, error)) return false;
    if (redirect.empty()) {
      *out = std::move(attempt);
      return true;
    }
    // Relative redirects are resolved against the current URL's origin.
    if (redirect.rfind("http://", 0) != 0 && redirect.rfind("https://", 0) != 0) {
      std::string host, path;
      int port = 80;
      if (!ParseHttpUrl(current, &host, &port, &path, error)) return false;
      const std::string origin =
          "http://" + host + (port == 80 ? "" : ":" + std::to_string(port));
      redirect = redirect[0] == '/' ? origin + redirect : origin + "/" + redirect;
    }
    current = redirect;
  }
  return Err(error, "too many redirects starting at " + url);
}

}  // namespace cs2mv
