cmake_minimum_required(VERSION 3.14)
project(SIM_SM VERSION 1.0)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies via FetchContent
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)


set(ARCHITECTURE_SOURCES
    src/architecture/gpu.cpp
    src/architecture/sm.cpp
    src/architecture/grid.cpp
    src/architecture/thread_block.cpp
    src/architecture/warp.cpp
    src/architecture/thread.cpp
    src/architecture/kernel.cpp
    src/architecture/flat_memory.cpp
    src/architecture/register_file.cpp
    src/architecture/occupancy.cpp
)

set(MEMORY_SOURCES
    src/memory/lru_policy.cpp
    src/memory/fifo_policy.cpp
    src/memory/random_policy.cpp
    src/memory/cache.cpp
    src/memory/memory_system.cpp
    src/memory/memory_coalescer.cpp
)

set(EXECUTION_SOURCES
    src/execution/instruction_executor.cpp
)

set(SCHEDULING_SOURCES
    src/scheduling/greedy_scheduler.cpp
    src/scheduling/round_robin_scheduler.cpp
    src/scheduling/priority_scheduler.cpp
)

set(RUNTIME_SOURCES
    src/runtime/config.cpp
    src/runtime/benchmarks.cpp
    src/runtime/performance_counter.cpp
)

set(CORE_SOURCES
    ${ARCHITECTURE_SOURCES}
    ${EXECUTION_SOURCES}
    ${SCHEDULING_SOURCES}
    ${RUNTIME_SOURCES}
    ${MEMORY_SOURCES}
)

# Core Library
add_library(sim_sm_core ${CORE_SOURCES})
target_include_directories(sim_sm_core PUBLIC include)
target_link_libraries(sim_sm_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(sim_sm_core PRIVATE -Wall -Wextra -Wpedantic)

# Main Executable
add_executable(gpu-sim src/main.cpp)
target_link_libraries(gpu-sim PRIVATE sim_sm_core)
target_compile_options(gpu-sim PRIVATE -Wall -Wextra -Wpedantic)

# Tests
add_executable(gpu-sim-tests tests/architecture_test.cpp tests/execution_test.cpp tests/scheduling_test.cpp tests/memory_test.cpp tests/memory_coalescer_test.cpp)
target_link_libraries(gpu-sim-tests PRIVATE sim_sm_core gtest_main)
target_compile_options(gpu-sim-tests PRIVATE -Wall -Wextra -Wpedantic)

include(GoogleTest)
gtest_discover_tests(gpu-sim-tests)
