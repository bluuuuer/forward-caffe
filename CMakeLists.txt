cmake_minimum_required(VERSION 2.8)

project(jaffe)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

include(${PROJECT_SOURCE_DIR}/cmake/Ex.cmake)

option(CPU_ONLY "Build Jaffe without CUDA support" OFF)

configure_file(cmake/jaffe_config.h.in 
		"${PROJECT_SOURCE_DIR}/include/cmake/jaffe_config.h")

include_directories(
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/eigen
		/home/xw/opencv3/include
		/usr/local/cuda/include/
)

link_directories(
		${PROJECT_SOURCE_DIR}/lib
		/home/xw/opencv3/lib
		/usr/local/cuda/lib64
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(cmake)
