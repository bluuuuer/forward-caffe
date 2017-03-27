macro(jaffe_cuda_compile objlist_variable)
  	foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    	set(${var}_backup_in_cuda_compile_ "${${var}}")
  	endforeach()

  	list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")

  	cuda_compile(cuda_objcs ${ARGN})

  	foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    	set(${var} "${${var}_backup_in_cuda_compile_}")
    	unset(${var}_backup_in_cuda_compile_)
  	endforeach()

  	set(${objlist_variable} ${cuda_objcs})
endmacro()
