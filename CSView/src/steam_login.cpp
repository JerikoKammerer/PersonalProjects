#include "cs2mv/steam_login.h"

#include <cstdio>
#include <cstring>

namespace cs2mv {
namespace {

FILE* OpenPipe(const std::string& command) {
#ifdef _WIN32
  return ::_popen(command.c_str(), "r");
#else
  return ::popen(command.c_str(), "r");
#endif
}

void ClosePipe(FILE* pipe) {
#ifdef _WIN32
  ::_pclose(pipe);
#else
  ::pclose(pipe);
#endif
}

std::string TrimRight(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
    s.pop_back();
  }
  return s;
}

// Runs a helper subcommand to completion and returns its first line.
bool RunLine(const std::string& command, std::string* line, std::string* error) {
  if (command.empty()) {
    if (error != nullptr) *error = "no game coordinator helper configured";
    return false;
  }
  FILE* pipe = OpenPipe(command + " 2>&1");
  if (pipe == nullptr) {
    if (error != nullptr) *error = "cannot run helper: " + command;
    return false;
  }
  std::string output;
  char buffer[1024];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
    if (output.size() > (1u << 16)) break;
  }
  ClosePipe(pipe);

  const std::size_t newline = output.find('\n');
  *line = TrimRight(newline == std::string::npos ? output : output.substr(0, newline));
  if (line->empty()) {
    if (error != nullptr) *error = "the helper said nothing";
    return false;
  }
  return true;
}

}  // namespace

const char* SteamLogin::StateName(State state) {
  switch (state) {
    case State::kIdle: return "idle";
    case State::kStarting: return "starting";
    case State::kWaiting: return "waiting";
    case State::kScanned: return "scanned";
    case State::kDone: return "done";
    case State::kError: return "error";
  }
  return "idle";
}

SteamLogin::~SteamLogin() { Cancel(); }

bool SteamLogin::Start(const std::string& helper_command, std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    if (error != nullptr) *error = "a sign-in is already in progress";
    return false;
  }
  if (helper_command.empty()) {
    if (error != nullptr) {
      *error =
          "no Steam helper is configured. Start the server with --gc-helper, "
          "or set CS2MV_GC_HELPER.";
    }
    return false;
  }

  if (worker_.joinable()) worker_.join();  // reap the previous attempt
  command_ = helper_command + " login --stream";
  snapshot_ = Snapshot();
  snapshot_.state = State::kStarting;
  running_ = true;
  worker_ = std::thread(&SteamLogin::Follow, this);
  return true;
}

// Reads the helper's event lines until it exits. Each line replaces the
// snapshot the page polls.
void SteamLogin::Follow() {
  std::string command;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    command = command_;
  }

  FILE* pipe = OpenPipe(command);
  if (pipe == nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.state = State::kError;
    snapshot_.message = "could not start the Steam helper";
    running_ = false;
    return;
  }

  // A QR data URL is a few kilobytes, so the line buffer has to be generous.
  std::string line;
  line.reserve(16384);
  char buffer[4096];
  bool saw_terminal_event = false;

  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    line += buffer;
    if (line.empty() || line.back() != '\n') continue;  // partial line
    const std::string event = TrimRight(line);
    line.clear();
    if (event.empty()) continue;

    std::lock_guard<std::mutex> lock(mutex_);
    if (event.compare(0, 3, "QR ") == 0) {
      snapshot_.qr_png = event.substr(3);
      snapshot_.state = State::kWaiting;
    } else if (event.compare(0, 4, "URL ") == 0) {
      snapshot_.qr_url = event.substr(4);
    } else if (event == "SCANNED") {
      snapshot_.state = State::kScanned;
    } else if (event.compare(0, 5, "DONE ") == 0) {
      snapshot_.account = event.substr(5);
      snapshot_.state = State::kDone;
      saw_terminal_event = true;
    } else if (event.compare(0, 6, "ERROR ") == 0) {
      snapshot_.message = event.substr(6);
      snapshot_.state = State::kError;
      saw_terminal_event = true;
    }
  }
  ClosePipe(pipe);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!saw_terminal_event && snapshot_.state != State::kDone) {
    snapshot_.state = State::kError;
    if (snapshot_.message.empty()) {
      snapshot_.message = "the Steam helper stopped without signing in";
    }
  }
  running_ = false;
}

SteamLogin::Snapshot SteamLogin::Get() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return snapshot_;
}

void SteamLogin::Cancel() {
  std::thread victim;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_.joinable()) victim = std::move(worker_);
    running_ = false;
  }
  // The helper exits on its own when the QR expires; detaching avoids blocking
  // the request thread on that.
  if (victim.joinable()) victim.detach();
}

bool SteamStatus(const std::string& helper_command, bool* signed_in,
                 std::string* account, std::string* error) {
  std::string line;
  if (!RunLine(helper_command + " status", &line, error)) return false;
  if (line.compare(0, 8, "SIGNEDIN") == 0) {
    *signed_in = true;
    *account = line.size() > 9 ? line.substr(9) : std::string();
    return true;
  }
  if (line.compare(0, 9, "SIGNEDOUT") == 0) {
    *signed_in = false;
    account->clear();
    return true;
  }
  if (error != nullptr) *error = "unexpected helper reply: " + line;
  return false;
}

bool SteamLogout(const std::string& helper_command, std::string* error) {
  std::string line;
  if (!RunLine(helper_command + " logout", &line, error)) return false;
  if (line.compare(0, 9, "SIGNEDOUT") == 0) return true;
  if (error != nullptr) *error = "unexpected helper reply: " + line;
  return false;
}

}  // namespace cs2mv
