cmake_minimum_required(VERSION 3.10)
project(utils)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")
endif()

option(ENABLE_DEBUG_INFO "Enable GDB debug information" OFF)

if(ENABLE_DEBUG_INFO)
    message(STATUS "GDB debug info enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

add_executable(calc_recall calc_recall.cpp)
add_executable(calc_incr_recall calc_incr_recall.cpp)
add_executable(compute_gt compute_gt.cpp)
add_executable(crop crop.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(calc_recall PRIVATE -Wall -Wextra)
    target_compile_options(calc_incr_recall PRIVATE -Wall -Wextra)
    target_compile_options(compute_gt PRIVATE -Wall -Wextra)
    target_compile_options(crop PRIVATE -Wall -Wextra)
    target_compile_options(fvecs_to_bin PRIVATE -Wall -Wextra)
endif()

install(TARGETS calc_recall calc_incr_recall compute_gt crop fvecs_to_bin
        RUNTIME DESTINATION bin)
