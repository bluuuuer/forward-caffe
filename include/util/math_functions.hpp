// huangshize 2016.03.16
// === math_function.h ===

#ifndef JAFFE_MATH_FUNCTIONS_H_
#define JAFFE_MATH_FUNCTIONS_H_
#include <string.h> 
#include <Eigen/Dense>

#include "common.h"

using namespace Eigen;

namespace jaffe {

    template <typename Dtype>
    void JaffeGemm(const int TransA, const int TranB, const int M, const int N,
		const int K, const Dtype alpha, const Dtype* A, const Dtype* B, 
		const Dtype beta, Dtype* C);

    template <typename Dtype>
    void JaffeGemv(const int TransA, const int M, const int N, const Dtype alpha, 
		const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

	template <typename Dtype>
	void JaffeCopy(const int N, const Dtype* X, Dtype* Y);

	template <typename Dtype>
	void JaffeSet(const int N, const Dtype alpha, Dtype* Y);

    template <typename Dtype>
    void JaffeAxpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

    template <typename Dtype>
    void JaffeSqr(const int N, const Dtype* a, Dtype *y);

	template <typename Dtype>
	void JaffePowx(const int N, const Dtype* a, const Dtype b, Dtype* y);

	template <typename Dtype>
	void JaffeMul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	void JaffeScal(const int N, const Dtype alpha, Dtype* x);

	template <typename Dtype>
	void JaffeAddScalar(const int N, const Dtype alpha, Dtype* y);

	template <typename Dtype>
	void JaffeExp(const int N, const Dtype* a, Dtype* y);

	template <typename Dtype>
	void JaffeDiv(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	template <typename Dtype>
	Dtype JaffeStridedDot(const int N, const Dtype* x, const int incx, 
		const Dtype* y, const int incy);

	template <typename Dtype>
	Dtype JaffeDot(const int N, const Dtype* x, const Dtype* y);

	inline void JaffeMemset(const size_t N, const int alpha, void* X) {
		memset(X, alpha, N);
	}

#ifndef CPU_ONLY // GPU

	template <typename Dtype>
	void JaffeGpuGemm(const int TransA, const int TransB, const int M, const int N, 
			const int K, const Dtype alpha, const Dtype* A, const Dtype* B, 
			const Dtype beta, Dtype* C);

	template <typename Dtype>
	void JaffeGpuGemv(const int TransA, const int M, const int N, const Dtype alpha,
			const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

	void JaffeGpuMemcpy(const size_t N, const void* X, void* Y);

	void JaffeGpu2CpuMemcpy(const size_t N, const void* X, void* Y);

	void JaffeGpuMemset(const size_t N, const int alpha, void* X);

	template <typename Dtype>
	void JaffeGpuDot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

	template <typename Dtype>
	void JaffeGpuAxpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

	// 向量相乘
	template <typename Dtype>
	void JaffeGpuMul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	// 求幂次
	template <typename Dtype>
	void JaffeGpuPowx(const int n, const Dtype* a, const Dtype alpha, Dtype* y);

	// 常量加法
	template <typename Dtype>
	void JaffeGpuAddScalar(const int N, const Dtype alpha, Dtype* y);

	// 常量除法
	template <typename Dtype>
	void JaffeGpuDiv(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	// 矩阵加法
	template <typename Dtype>
	void JaffeGpuAdd(const int N, const Dtype alpha, const Dtype* a, const Dtype* b, 
		Dtype* y);

	// 矩阵与元素相乘
	template <typename Dtype>
	void JaffeGpuScale(const int N, const Dtype alpha, const Dtype* x, Dtype* y);

#endif // GPU
}
#endif
