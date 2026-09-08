#include <string>

#include "cs2mv/http_client.h"
#include "cs2mv/http_server.h"
#include "cs2mv/locator.h"
#include "test_util.h"

using cs2mv::HttpServer;
using cs2mv::ParseHttpUrl;

TEST(UrlParsingSplitsHostPortPath) {
  std::string host, path, error;
  int port = 0;

  CHECK(ParseHttpUrl("http://replay191.valve.net/730/003693.dem.bz2", &host,
                     &port, &path, &error));
  CHECK_EQ(host, std::string("replay191.valve.net"));
  CHECK_EQ(port, 80);
  CHECK_EQ(path, std::string("/730/003693.dem.bz2"));

  CHECK(ParseHttpUrl("http://localhost:8080/api/match?code=x", &host, &port,
                     &path, &error));
  CHECK_EQ(host, std::string("localhost"));
  CHECK_EQ(port, 8080);
  CHECK_EQ(path, std::string("/api/match?code=x"));

  CHECK(ParseHttpUrl("http://example.com", &host, &port, &path, &error));
  CHECK_EQ(path, std::string("/"));
}

TEST(UrlParsingRejectsUnsupportedSchemes) {
  std::string host, path, error;
  int port = 0;
  CHECK(!ParseHttpUrl("https://example.com/x", &host, &port, &path, &error));
  CHECK(error.find("https") != std::string::npos);
  CHECK(!ParseHttpUrl("ftp://example.com/x", &host, &port, &path, &error));
  CHECK(!ParseHttpUrl("/just/a/path", &host, &port, &path, &error));
  CHECK(!ParseHttpUrl("http://example.com:99999/x", &host, &port, &path, &error));
}

TEST(UrlDecodeHandlesEscapes) {
  CHECK_EQ(HttpServer::UrlDecode("CSGO-a%2Db"), std::string("CSGO-a-b"));
  CHECK_EQ(HttpServer::UrlDecode("a+b"), std::string("a b"));
  CHECK_EQ(HttpServer::UrlDecode("C%3A%5Cdemos%5Cmatch.dem"),
           std::string("C:\\demos\\match.dem"));
  // A stray percent is passed through rather than swallowing later characters.
  CHECK_EQ(HttpServer::UrlDecode("100%"), std::string("100%"));
  CHECK_EQ(HttpServer::UrlDecode("%zz"), std::string("%zz"));
}

TEST(DemoIndexRoundTrips) {
  const std::string path = testing::DataDir() + "/test_index.txt";

  cs2mv::DemoIndex index;
  std::string error;
  index.Set("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA",
            "http://replay191.valve.net/730/003693.dem.bz2");
  index.Set("3253932207440199901", "C:\\demos\\match.dem");
  CHECK(index.Save(path, &error));

  cs2mv::DemoIndex loaded;
  CHECK(loaded.Load(path, &error));
  CHECK_EQ(loaded.entries().size(), static_cast<std::size_t>(2));

  cs2mv::DemoLocation location;
  CHECK(loaded.Lookup("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA", &location));
  CHECK_EQ(static_cast<int>(location.kind), static_cast<int>(cs2mv::DemoLocation::kUrl));

  CHECK(loaded.Lookup("3253932207440199901", &location));
  CHECK_EQ(static_cast<int>(location.kind), static_cast<int>(cs2mv::DemoLocation::kFile));
  CHECK_EQ(location.value, std::string("C:\\demos\\match.dem"));

  CHECK(!loaded.Lookup("nothing here", &location));
  std::remove(path.c_str());
}

TEST(MissingIndexIsNotAnError) {
  cs2mv::DemoIndex index;
  std::string error;
  CHECK(index.Load(testing::DataDir() + "/definitely_not_here.txt", &error));
  CHECK(index.entries().empty());
}

TEST(ShareCodeExtractedFromSurroundingText) {
  // What CS2's copy button actually puts on the clipboard.
  const std::string steam_url =
      "steam://rungame/730/76561202255233023/+csgo_download_match%20"
      "CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA";
  std::string code;
  CHECK(cs2mv::ExtractShareCode(steam_url, &code));
  CHECK_EQ(code, std::string("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA"));

  CHECK(cs2mv::ExtractShareCode(
      "gg wp here it is CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA have a look", &code));
  CHECK_EQ(code, std::string("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA"));

  // A bare code, and a bare payload with no prefix.
  CHECK(cs2mv::ExtractShareCode("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA", &code));
  CHECK(cs2mv::ExtractShareCode("Cji4ZrQMJJs6JyqovwoSmJkDA", &code));

  CHECK(!cs2mv::ExtractShareCode("no code in here at all", &code));
  CHECK(!cs2mv::ExtractShareCode("CSGO-tooshort", &code));
}

TEST(ReplayDirectoryLookupIsWellFormed) {
  // Whether CS2 is installed is not something a test can assume, so this only
  // checks the contract: every path returned exists and ends in "replays".
  for (const std::string& dir : cs2mv::Cs2ReplayDirectories()) {
    CHECK(dir.size() > 7);
    CHECK(dir.find("replays") != std::string::npos);
  }
  // A match id that cannot exist must not be claimed as found.
  cs2mv::ShareCode nonsense;
  nonsense.match_id = 1;
  nonsense.outcome_id = 2;
  std::string path = "untouched";
  CHECK(!cs2mv::FindDownloadedDemo(nonsense, &path));
  CHECK_EQ(path, std::string("untouched"));
}

TEST(ResolveExplainsWhatIsMissing) {
  cs2mv::ShareCode code;
  CHECK(cs2mv::DecodeShareCode("CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA", &code, nullptr));

  cs2mv::ResolveOptions options;
  options.cache_dir = testing::DataDir() + "/empty_cache";
  options.index_path = testing::DataDir() + "/empty_cache/index.txt";
  options.allow_download = false;

  std::string demo_path, error;
  CHECK(!cs2mv::ResolveDemo(code, "CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA", options,
                            &demo_path, &error));
  // The message has to tell the user what to do next, not just fail. The
  // useful first move is downloading the demo in CS2, not editing an index.
  CHECK(error.find("Watch -> Your Matches") != std::string::npos);
  CHECK(error.find("cs2mv add") != std::string::npos);
  CHECK(error.find(std::to_string(code.match_id)) != std::string::npos);
}
