cmake_minimum_required(VERSION 3.21)
project(power_rune)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(power_rune main.cpp)

target_link_libraries(power_rune ${OpenCV_LIBS})