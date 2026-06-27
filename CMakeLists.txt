cmake_minimum_required(VERSION 3.10)
project(AlienLLM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -mavx2 -mfma -ffast-math)
endif()

# Add include directory
include_directories(include)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

