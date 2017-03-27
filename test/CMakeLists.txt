project(jaffe)

find_package(CUDA QUIET REQUIRED)

list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")

#add_executable(jaffe_test main.cpp)
add_executable(classification_test classification.cpp)
#add_executable(extract extract_bottom_top.cpp)
#add_executable(l2norm_test l2norm_test.cpp)
cuda_add_executable(l2norm_test_gpu l2norm_gpu_test.cpp)
#add_executable(cuda_test cuda_test.cpp)

#target_link_libraries(jaffe_test jaffe_shared)
target_link_libraries(classification_test jaffe)
#target_link_libraries(extract jaffe_shared)
#target_link_libraries(l2norm_test jaffe)
target_link_libraries(l2norm_test_gpu jaffe)
#target_link_libraries(cuda_test jaffe)
