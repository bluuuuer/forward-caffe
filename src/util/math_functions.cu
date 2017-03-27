#include <math_functions.h> // CUDA's
#include <cmath>

#include "util/math_functions.hpp"

namespace jaffe {
	template <>
	void JaffeGpuGemm<float>(const int TransA, const int TransB, const int M, 
			const int N, const int K, const float alpha, const float* A, 
			const float* B, const float beta, float* C) {
		//cout << "JaffeGpuGemm<float>" << endl;
		int lda = (TransA == 0) ? K : M;
		int ldb = (TransB == 0) ? N : K;
		cublasOperation_t cuTransA = (TransA == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB = (TransB == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasSgemm(Jaffe::GetCublasHandle(), cuTransB, cuTransA, N, M, 
			K, &alpha, B, ldb, A, lda, &beta, C, N));
	}

	template <>
	void JaffeGpuGemm<double>(const int TransA, const int TransB, const int M, 
			const int N, const int K, const double alpha, const double* A, 
			const double* B, const double beta, double* C) {
		int lda = (TransA == 0) ? K : M;
		int ldb = (TransB == 0) ? N : K;
		cublasOperation_t cuTransA = (TransA == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t cuTransB = (TransB == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
		CUBLAS_CHECK(cublasDgemm(Jaffe::GetCublasHandle(), cuTransB, cuTransA, N, M, K,
			&alpha, B, ldb, A, lda, &beta, C, N));
	}

	template <>
	void JaffeGpuGemv<float>(const int TransA, const int M, const int N, 
			const float alpha, const float* A, const float* x, const float beta,
			float* y) {
		cublasOperation_t cuTransA = (TransA == 0) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK(cublasSgemv(Jaffe::GetCublasHandle(), cuTransA, N, M, &alpha, A,
			N, x, 1, &beta, y, 1));
	}

	template <>
	void JaffeGpuGemv<double>(const int TransA, const int M, const int N, 
			const double alpha, const double* A, const double* x, const double beta,
			double* y) {
		cublasOperation_t cuTransA = (TransA == 0) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK(cublasDgemv(Jaffe::GetCublasHandle(), cuTransA, N, M, &alpha, A,
			N, x, 1, &beta, y, 1));
	}

	// X->Y
	void JaffeGpuMemcpy(const size_t N, const void* X, void* Y) {
		if (X != Y) {
			CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
		}
	}

	void JaffeGpu2CpuMemcpy(const size_t N, const void* X, void* Y) {
		if (X != Y) {
			CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDeviceToHost));	
		}
	}

	void JaffeGpuMemset(const size_t N, const int alpha, void* X) {
		CUDA_CHECK(cudaMemset(X, alpha, N));
	}

	template <>
	void JaffeGpuDot<float>(const int n, const float* x, const float* y, float* out) {
		CUBLAS_CHECK(cublasSdot(Jaffe::GetCublasHandle(), n, x, 1, y, 1, out));	
	}
	
	template <>
	void JaffeGpuDot<double>(const int n, const double* x, const double* y,
					double* out) {
		CUBLAS_CHECK(cublasDdot(Jaffe::GetCublasHandle(), n, x, 1, y, 1, out));
	}

	template <>
	void JaffeGpuAxpy<float>(const int N, const float alpha, const float* X, 
					float* Y) {
		CUBLAS_CHECK(cublasSaxpy(Jaffe::GetCublasHandle(), N, &alpha, X, 1, Y, 1));
	}

	template <>
	void JaffeGpuAxpy<double>(const int N, const double alpha, const double* X,
					double* Y) {
		CUBLAS_CHECK(cublasDaxpy(Jaffe::GetCublasHandle(), N, &alpha, X, 1, Y, 1));
	}

	template <typename Dtype>
	__global__ void mul_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = a[index] * b[index];
		}
	}

	template <>
	void JaffeGpuMul<float>(const int N, const float* a, const float* b, float* y) {
		mul_kernel<float><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, a, b, 
			y);	
	}

	template <>
	void JaffeGpuMul<double>(const int N, const double* a, const double* b, 
					double* y) {
		mul_kernel<double><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, a, b, 
			y);		
	}

	template <typename Dtype>
	__global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, 
					Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = pow(a[index], alpha);
		}
	}

	template <>
	void JaffeGpuPowx<float>(const int n, const float* a, const float alpha, 
					float* y) {
		powx_kernel<float><<<JAFFE_GET_BLOCKS(n), JAFFE_CUDA_NUM_THREADS>>>(n, a, 
			alpha, y);
	}

	template <>
	void JaffeGpuPowx<double>(const int n, const double* a, const double alpha, 
					double* y) {
		powx_kernel<double><<<JAFFE_GET_BLOCKS(n), JAFFE_CUDA_NUM_THREADS>>>(n, a, 
			alpha, y);
	}

	template <typename Dtype>
	__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] += alpha;
		}			
	}

	template <>
	void JaffeGpuAddScalar<float>(const int N, const float alpha, float* Y) {
		add_scalar_kernel<float><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N,
			alpha, Y);
	}

	template <>
	void JaffeGpuAddScalar<double>(const int N, const double alpha, double* Y) {
		add_scalar_kernel<double><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N,
			alpha, Y);
	}

	template <typename Dtype>
	__global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, 
					Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = a[index] / b[index];
		}	
	}

	template <>
	void JaffeGpuDiv<float>(const int N, const float* a, const float* b, float* y) {
		div_kernel<float><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, a, b, 
			y);
	}

	template <>
	void JaffeGpuDiv<double>(const int N, const double* a, const double* b, double* y) {
		div_kernel<double><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, a, b, 
			y);
	}

	template <typename Dtype>
	__global__ void add_kernel(const int n, const Dtype alpha, const Dtype* a, 
			const Dtype* b, Dtype* y) {
		CUDA_KERNEL_LOOP(index, n) {
			y[index] = alpha * a[index] + b[index];
		}
	}

	template<>
	void JaffeGpuAdd<float>(const int N, const float alpha, const float* a, 
			const float* b, float* y) {
		add_kernel<float><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, alpha, 
			a, b, y);		
	}

	template<>
	void JaffeGpuAdd<double>(const int N, const double alpha, const double* a, 
			const double* b, double* y) {
		add_kernel<double><<<JAFFE_GET_BLOCKS(N), JAFFE_CUDA_NUM_THREADS>>>(N, alpha, 
			a, b, y);		
	}

	template<>
	void JaffeGpuScale<float>(const int n, const float alpha, const float* x, 
					float* y) {
		CUBLAS_CHECK(cublasScopy(Jaffe::GetCublasHandle(), n, x, 1, y, 1));
		CUBLAS_CHECK(cublasSscal(Jaffe::GetCublasHandle(), n, &alpha, y, 1));	
	}

	template<>
	void JaffeGpuScale<double>(const int n, const double alpha, const double* x, 
					double* y) {
		CUBLAS_CHECK(cublasDcopy(Jaffe::GetCublasHandle(), n, x, 1, y, 1));
		CUBLAS_CHECK(cublasDscal(Jaffe::GetCublasHandle(), n, &alpha, y, 1));	
	}
	
} // namespace jaffe
