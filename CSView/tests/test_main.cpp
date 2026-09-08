#include <cstdio>
#include <cstring>
#include <string>

#include "test_util.h"

int main(int argc, char** argv) {
  if (argc > 1) testing::DataDir() = argv[1];

  const char* filter = argc > 2 ? argv[2] : nullptr;
  int ran = 0;
  for (auto& test : testing::Registry()) {
    if (filter != nullptr && test.name.find(filter) == std::string::npos) continue;
    testing::CurrentTest() = test.name;
    const int before = testing::FailureCount();
    test.body();
    ++ran;
    if (testing::FailureCount() == before) {
      std::printf("  ok   %s\n", test.name.c_str());
    }
  }

  const int failures = testing::FailureCount();
  std::printf("\n%d test%s, %d failure%s\n", ran, ran == 1 ? "" : "s", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
