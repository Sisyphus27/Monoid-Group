cmake_minimum_required(VERSION 3.22)
project(cpp)

SET(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 14)

add_executable(cpp main.cpp Ssort.cpp Ssort.h utils.h)