// Thin platform shim so the HTTP client and server can share one spelling of
// sockets. Internal to src/.
#ifndef CS2MV_SRC_NET_H_
#define CS2MV_SRC_NET_H_

#include <string>

#ifdef _WIN32
// Keep <windows.h> from dragging in winsock 1.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace cs2mv {
namespace net {

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kInvalidSocket = INVALID_SOCKET;
inline void CloseSocket(Socket s) { ::closesocket(s); }
inline int LastError() { return ::WSAGetLastError(); }
#else
using Socket = int;
constexpr Socket kInvalidSocket = -1;
inline void CloseSocket(Socket s) { ::close(s); }
inline int LastError() { return errno; }
#endif

// Winsock needs an explicit startup call; this makes it happen once, from
// wherever the first socket is created. A no-op elsewhere.
class Startup {
 public:
  Startup() {
#ifdef _WIN32
    WSADATA data;
    ok_ = WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
    ok_ = true;
#endif
  }
  ~Startup() {
#ifdef _WIN32
    if (ok_) WSACleanup();
#endif
  }
  bool ok() const { return ok_; }

 private:
  bool ok_ = false;
};

// Ensures winsock is initialised for the lifetime of the process.
inline bool EnsureInitialised() {
  static Startup startup;
  return startup.ok();
}

inline bool SendAll(Socket s, const char* data, std::size_t size) {
  std::size_t sent = 0;
  while (sent < size) {
    const int n = ::send(s, data + sent, static_cast<int>(size - sent), 0);
    if (n <= 0) return false;
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

inline bool SendAll(Socket s, const std::string& data) {
  return SendAll(s, data.data(), data.size());
}

}  // namespace net
}  // namespace cs2mv

#endif  // CS2MV_SRC_NET_H_
