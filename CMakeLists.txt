cmake_minimum_required (VERSION 2.6)
project(WibergMinimization)

set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")

include_directories("../.." "${PROJECT_SOURCE_DIR}")
add_subdirectory(l1)
add_subdirectory(multiple_instance)
add_subdirectory(utility)
