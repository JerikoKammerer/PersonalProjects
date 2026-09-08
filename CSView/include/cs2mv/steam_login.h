// Browser-driven Steam sign-in, for the game coordinator helper.
//
// The only thing that crosses this boundary is a QR challenge: the browser
// shows it, the phone scans it and talks to Steam directly, and the helper
// writes a refresh token to its own config. No password is entered anywhere,
// and neither this process nor the page ever holds a credential - which is why
// putting this in a web page is reasonable at all.
//
// The flow is asynchronous, so it does not fit the run-and-read-a-line shape of
// RunGcHelper: the helper is started, its stdout is followed on a background
// thread, and the page polls for progress.
#ifndef CS2MV_STEAM_LOGIN_H_
#define CS2MV_STEAM_LOGIN_H_

#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace cs2mv {

class SteamLogin {
 public:
  enum class State {
    kIdle,        // never started, or reset
    kStarting,    // helper launched, no QR yet
    kWaiting,     // QR is available, waiting for a phone
    kScanned,     // scanned, waiting for approval
    kDone,        // signed in; a refresh token now exists
    kError,
  };

  ~SteamLogin();

  // Launches `helper_command login --stream`. Returns false if one is already
  // in flight or the helper could not be started.
  bool Start(const std::string& helper_command, std::string* error);

  struct Snapshot {
    State state = State::kIdle;
    std::string qr_png;    // data:image/png;base64,... ready for an <img src>
    std::string qr_url;    // the same challenge as a link
    std::string account;
    std::string message;   // set when state is kError
  };
  Snapshot Get() const;

  // Abandons an in-flight attempt. Safe to call at any time.
  void Cancel();

  static const char* StateName(State state);

 private:
  void Follow();

  mutable std::mutex mutex_;
  Snapshot snapshot_;
  std::string command_;
  std::thread worker_;
  bool running_ = false;
};

// Asks the helper whether a refresh token exists. `account` is filled with the
// signed-in account name when there is one. Returns false when the helper could
// not be run at all.
bool SteamStatus(const std::string& helper_command, bool* signed_in,
                 std::string* account, std::string* error);

// Removes the stored refresh token.
bool SteamLogout(const std::string& helper_command, std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_STEAM_LOGIN_H_
