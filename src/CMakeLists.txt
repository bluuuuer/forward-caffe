aux_source_directory(. srcs)
file(GLOB_RECURSE srcs *.cpp *.cc)

find_package(OpenCV HINTS /home/xw/opencv3)
find_library(PROTO_BUF_LIBS protobuf /home/xw/Project/forward-caffe/lib)

list(APPEND srcs ${PROJECT_SOURCE_DIR}/cmake/jaffe_config.h)

if (${CPU_ONLY} STREQUAL "OFF")
	find_package(CUDA QUIET REQUIRED)
	include(FindCUDA)

	file(GLOB_RECURSE cuda ${PROJECT_SOURCE_DIR}/src/*.cu)
	#	list(APPEND cuda ${PROJECT_SOURCE_DIR}/include/*.hpp)
	#list(APPEND cuda ${PROJECT_SOURCE_DIR}/include/*.h)
	list(APPEND cuda ${PROJECT_SOURCE_DIR}/cmake/jaffe_config.h)
	
	#message(${cuda})
	
	jaffe_cuda_compile(cuda_objs ${cuda})

	#message(${cuda_objs})

	list(APPEND srcs ${cuda_objs} ${cuda})
endif()

#message(${srcs})

add_library(jaffe SHARED ${srcs})

target_link_libraries(jaffe ${OpenCV_LIBS} ${PROTO_BUF_LIBS})

if (${CPU_ONLY} STREQUAL "OFF")
		cuda_add_cublas_to_target(jaffe)
	target_link_libraries(jaffe ${CUDA_LIBRARIES})
endif()

set_target_properties(jaffe PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(jaffe PROPERTIES CLEAN_DIRECT_OUTPUT 1)
