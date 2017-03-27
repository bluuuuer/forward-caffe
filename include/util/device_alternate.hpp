#ifndef JAFFE_UTIL_DEVICE_ALTERNATE_HPP
#define JAFFE_UTIL_DEVICE_ALTERNATE_HPP

#include <iostream>
#include "cmake/jaffe_config.h"

using std::cout;
using std::endl;

#ifdef CPU_ONLY  // CPU-only Caffe.

#define NO_GPU cout << "Cannot use GPU in CPU-only Caffe: check mode." << endl

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom, \
	const vector<JBlob<Dtype>*>& top) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types

static void HandleError(cudaError_t error, const char * file, int line) {
	if (error != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString(error), file, line );
		exit( EXIT_FAILURE );		
	}
}

#define CUDA_CHECK(condition) (HandleError(condition , __FILE__, __LINE__ ))
  /* Code block avoids redefinition of cudaError_t error */ \
		  /*
  	do { \
    	cudaError_t error = condition; \
		if (error != cudaSuccess) { \
			cout << cudaGetErrorString(error) << endl; \
		} \
  	} while (0)
	*/

#define CUBLAS_CHECK(condition) \
  	do { \
    	cublasStatus_t status = condition; \
		if(status != CUBLAS_STATUS_SUCCESS) { \
			cout << jaffe::cublasGetErrorString(status) << endl; \
		} \
  	} while (0)

/*
#define CURAND_CHECK(condition) \
  	do { \
    	curandStatus_t status = condition; \
		if (status != CURAND_STATUS_SUCCESS) { \
			cout << jaffe::curandGetErrorString(status) << endl; \
		} \
  } while (0)
  */

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
		i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace jaffe {

// CUDA: library error reporting.
	const char* cublasGetErrorString(cublasStatus_t error);
	//const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
	const int JAFFE_CUDA_NUM_THREADS = 1024; 
// CUDA: number of blocks for threads.
	inline int JAFFE_GET_BLOCKS(const int N) {
  		return (N + JAFFE_CUDA_NUM_THREADS - 1) / JAFFE_CUDA_NUM_THREADS;
	}

}  // namespace jaffe

#endif  // CPU_ONLY

#endif  // JAFFE_UTIL_DEVICE_ALTERNATE_HPP
