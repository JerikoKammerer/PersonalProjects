cmake_minimum_required(VERSION 3.15)
project(cs2_match_viewer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN NOMINMAX)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Everything except the CLI entry point, so the tests can link against it.
add_library(cs2mv STATIC
  src/bzip2.cpp
  src/demo.cpp
  src/gc.cpp
  src/http_client.cpp
  src/http_server.cpp
  src/json.cpp
  src/locator.cpp
  src/match.cpp
  src/parser.cpp
  src/protobuf.cpp
  src/sharecode.cpp
  src/steam_login.cpp
  src/snappy.cpp
)
target_include_directories(cs2mv PUBLIC include PRIVATE src)

find_package(Threads REQUIRED)
target_link_libraries(cs2mv PUBLIC Threads::Threads)
if(WIN32)
  # ws2_32 for sockets, advapi32 for the registry lookup that finds Steam.
  target_link_libraries(cs2mv PUBLIC ws2_32 advapi32)
endif()

add_executable(cs2mv_cli src/main.cpp)
target_link_libraries(cs2mv_cli PRIVATE cs2mv)
set_target_properties(cs2mv_cli PROPERTIES OUTPUT_NAME cs2mv)

add_executable(cs2mv_tests
  tests/test_main.cpp
  tests/test_codecs.cpp
  tests/test_demo.cpp
  tests/test_http.cpp
  tests/test_protobuf.cpp
  tests/test_sharecode.cpp
)
target_link_libraries(cs2mv_tests PRIVATE cs2mv)
target_include_directories(cs2mv_tests PRIVATE tests)

enable_testing()
add_test(NAME cs2mv_tests
         COMMAND cs2mv_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

# The web UI is read from disk at run time; copy it next to the binary so the
# default --web path works from the build directory too.
add_custom_command(TARGET cs2mv_cli POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/web"
          "$<TARGET_FILE_DIR:cs2mv_cli>/web")
