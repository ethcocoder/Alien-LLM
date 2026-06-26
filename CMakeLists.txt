cmake_minimum_required(VERSION 3.10)
project(AlienLLM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(include)

# Find Eigen3 (for linear algebra)
# We might need to install it if not available
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Executable
add_executable(alien_llm ${SOURCES} src/main.cpp)

# Libraries
# add_library(alien_llm_lib STATIC ${SOURCES})

# Testing
enable_testing()
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})
# add_executable(run_tests tests/test_main.cpp)
# target_link_libraries(run_tests alien_llm_lib ${GTEST_LIBRARIES} pthread)
# add_test(NAME AllTests COMMAND run_tests)
