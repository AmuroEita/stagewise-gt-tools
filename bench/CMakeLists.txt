cmake_minimum_required(VERSION 3.10)
project(ANN-CC-bench)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(algorithms/hnsw/hnswlib hnswlib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(index SHARED
    algorithms/index_cgo.cpp
)

target_include_directories(index PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/algorithms
    ${CMAKE_SOURCE_DIR}/../hnswlib
)

target_link_libraries(index PRIVATE
    hnswlib
)