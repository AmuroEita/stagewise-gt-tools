cmake_minimum_required(VERSION 3.10)
project(ComputeGT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -mavx2 -mfma)
endif()

find_library(PAPI_LIBRARY NAMES papi libpapi
             HINTS /usr/lib /usr/local/lib /opt/local/lib
             DOC "PAPI library")
if(NOT PAPI_LIBRARY)
    message(FATAL_ERROR "PAPI library not found. Please install libpapi-dev (e.g., sudo apt-get install libpapi-dev)")
endif()

find_path(PAPI_INCLUDE_DIR NAMES papi.h
          HINTS /usr/include /usr/local/include /opt/local/include
          DOC "PAPI include directory")
if(NOT PAPI_INCLUDE_DIR)
    message(FATAL_ERROR "PAPI header (papi.h) not found. Please install PAPI development files.")
endif()

add_executable(compute_gt compute_gt.cpp)
add_executable(crop crop.cpp)
add_executable(check_recall check_recall.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)

# add_executable(cc_bench cc_bench.cpp)
# target_include_directories(cc_bench PRIVATE ${PAPI_INCLUDE_DIR})
# target_link_libraries(cc_bench PRIVATE ${PAPI_LIBRARY})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)