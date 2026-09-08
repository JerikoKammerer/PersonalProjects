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
  // The message has to tell the user what to do next, not just fail.
  CHECK(error.find("cs2mv add") != std::string::npos);
  CHECK(error.find(std::to_string(code.match_id)) != std::string::npos);
}
