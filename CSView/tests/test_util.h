// A tiny test harness: registration by static constructor, a handful of
// assertion macros, and a main() that runs everything.
#ifndef CS2MV_TESTS_TEST_UTIL_H_
#define CS2MV_TESTS_TEST_UTIL_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> registry;
  return registry;
}

inline int& FailureCount() {
  static int failures = 0;
  return failures;
}

inline std::string& CurrentTest() {
  static std::string name;
  return name;
}

inline void ReportFailure(const char* file, int line, const std::string& what) {
  ++FailureCount();
  std::fprintf(stderr, "  FAIL %s\n    %s:%d: %s\n", CurrentTest().c_str(), file,
               line, what.c_str());
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) {
    Registry().push_back(TestCase{name, std::move(body)});
  }
};

// The directory holding tests/data, set from argv[1] or a build-time default.
inline std::string& DataDir() {
  static std::string dir = "tests/data";
  return dir;
}

inline std::string ReadFileOrDie(const std::string& name) {
  const std::string path = DataDir() + "/" + name;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr,
                 "cannot open fixture %s\n"
                 "run: python tools/make_fixture.py, or pass the data directory "
                 "as the first argument\n",
                 path.c_str());
    std::exit(2);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

template <typename A, typename B>
std::string Describe(const char* expr, const A& got, const B& want) {
  std::ostringstream ss;
  ss << expr << "\n      got:  " << got << "\n      want: " << want;
  return ss.str();
}

}  // namespace testing

#define TEST(name)                                                       \
  static void Test_##name();                                             \
  static ::testing::Registrar kRegister_##name(#name, Test_##name);      \
  static void Test_##name()

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) ::testing::ReportFailure(__FILE__, __LINE__, #cond);    \
  } while (0)

#define CHECK_EQ(got, want)                                              \
  do {                                                                   \
    const auto got_value = (got);                                        \
    const auto want_value = (want);                                      \
    if (!(got_value == want_value)) {                                    \
      ::testing::ReportFailure(                                          \
          __FILE__, __LINE__,                                            \
          ::testing::Describe(#got " == " #want, got_value, want_value));\
    }                                                                    \
  } while (0)

#define CHECK_NEAR(got, want, tolerance)                                 \
  do {                                                                   \
    const double got_value = (got);                                      \
    const double want_value = (want);                                    \
    if (std::fabs(got_value - want_value) > (tolerance)) {               \
      ::testing::ReportFailure(                                          \
          __FILE__, __LINE__,                                            \
          ::testing::Describe(#got " ~= " #want, got_value, want_value)); \
    }                                                                    \
  } while (0)

#endif  // CS2MV_TESTS_TEST_UTIL_H_
