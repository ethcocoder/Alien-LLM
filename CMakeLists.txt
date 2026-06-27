cmake_minimum_required(VERSION 3.10)
project(AlienLLM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(include)

# Find Eigen3 (for linear algebra)
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Source files for the library
set(LIB_SOURCES src/bridge.cpp)

# Build shared library
add_library(alien_llm_lib SHARED ${LIB_SOURCES})
set_target_properties(alien_llm_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(alien_llm_lib PUBLIC Eigen3::Eigen)

# Executable for testing
add_executable(alien_llm src/main.cpp)
target_link_libraries(alien_llm PRIVATE Eigen3::Eigen)

# Checkpoint verification test
add_executable(test_checkpoint scratch/test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE Eigen3::Eigen)
