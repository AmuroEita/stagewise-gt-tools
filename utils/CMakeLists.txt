cmake_minimum_required(VERSION 3.10)
project(utils)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(check_recall check_recall.cpp)
add_executable(compute_gt compute_gt.cpp)
add_executable(crop crop.cpp)
add_executable(fvecs_to_bin fvecs_to_bin.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(check_recall PRIVATE -Wall -Wextra)
    target_compile_options(compute_gt PRIVATE -Wall -Wextra)
    target_compile_options(crop PRIVATE -Wall -Wextra)
    target_compile_options(fvecs_to_bin PRIVATE -Wall -Wextra)
endif()

install(TARGETS check_recall compute_gt crop fvecs_to_bin
        RUNTIME DESTINATION bin)
