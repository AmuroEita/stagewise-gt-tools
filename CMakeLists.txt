cmake_minimum_required(VERSION 3.10)
project(ComputeGT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -mavx2 -mfma)
endif()

add_executable(compute_gt compute_gt.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)