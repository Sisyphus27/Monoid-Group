cmake_minimum_required(VERSION 3.22)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-g")

add_executable(cpp main.cpp Ssort.cpp Ssort.h utils.cpp utils.h)