cmake_minimum_required(VERSION 3.10)
project(ANN-CC-bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)

add_subdirectory(algorithms/hnsw/hnswlib hnswlib)
add_subdirectory(algorithms/parlayann/parlayann parlayann)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(index SHARED
    algorithms/index_cgo.cpp
)

target_include_directories(index PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/algorithms
    ${CMAKE_SOURCE_DIR}/../hnswlib
    ${CMAKE_SOURCE_DIR}/../parlayann
)

target_link_libraries(index PRIVATE
    hnswlib
    ParlayANN
    OpenMP::OpenMP_CXX
)
