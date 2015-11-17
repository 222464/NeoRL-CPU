cmake_minimum_required(VERSION 2.8)

project(NeoRL-CPU)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g")
endif()

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
file(GLOB_RECURSE LINK_SRC
    "source/*.h"
    "source/*.cpp"
)
 
add_executable(NeoRL-CPU ${LINK_SRC})
