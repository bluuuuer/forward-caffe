// huangshize 2016.03.29 // === math_function.cpp ===

#include "util/math_functions.hpp"
#include "common.h"

namespace jaffe {
	
	template <>
	void JaffeGemm<float>(const int TransA, const int TransB, const int M, 
		const int N, const int K, const float alpha, const float* A, const float* B, 
		const float beta, float* C) {
		MatrixXf MA, MB, MC;
		if (TransA == 0)
			MA = Map<const MatrixXf, 0, Stride<1, Dynamic> > (A, M, K, 
				Stride<1, Dynamic>(1, K));
		else {
			MA = Map<const MatrixXf, 0, Stride<1, Dynamic> > (A, K, M, 
				Stride<1, Dynamic>(1, M));
			MA.transposeInPlace();
		}
		if (TransB == 0)
			MB = Map<const MatrixXf, 0, Stride<1, Dynamic> > (B, K, N, 
				Stride<1, Dynamic>(1, N));
		else {
			MB = Map<const MatrixXf, 0, Stride<1, Dynamic> > (B, N, K, 
				Stride<1, Dynamic>(1, K));
			MB.transposeInPlace();
		}
		if (beta)
			MC = Map<MatrixXf, 0, Stride<1, Dynamic> > (C, M, N, 
				Stride<1, Dynamic>(1, N));
		else 
			MC = MatrixXf::Zero(M, N);
		Map<MatrixXf, 0, Stride<1, Dynamic> > (C, M, N, Stride<1, Dynamic>(1, N)) 
				= alpha * MA * MB + MC;
	}

	template <>
	void JaffeGemm<double>(const int TransA, const int TransB, const int M, 
		const int N, const int K, const double alpha, const double* A, 
		const double* B, const double beta, double* C) {
		MatrixXd MA, MB, MC;
		if (TransA == 0)
			MA = Map<const MatrixXd, 0, Stride<1, Dynamic> > (A, M, K, 
				Stride<1, Dynamic>(1, K));
		else {
			MA = Map<const MatrixXd, 0, Stride<1, Dynamic> > (A, K, M, 
				Stride<1, Dynamic>(1, M));
			MA.transposeInPlace();
		}
		if (TransB == 0)
			MB = Map<const MatrixXd, 0, Stride<1, Dynamic> > (B, K, N, 
				Stride<1, Dynamic>(1, N));
		else {
			MB = Map<const MatrixXd, 0, Stride<1, Dynamic> > (B, N, K, 
				Stride<1, Dynamic>(1, K));
			MB.transposeInPlace();
		}
		if (beta)
			MC = Map<MatrixXd, 0, Stride<1, Dynamic> > (C, M, N, 
				Stride<1, Dynamic>(1, N));
		else 
			MC = MatrixXd::Zero(M, N);
		Map<MatrixXd, 0, Stride<1, Dynamic> > (C, M, N, Stride<1, Dynamic>(1, N)) 
				= alpha * MA * MB + MC;
	}

	template <>
	void JaffeGemv<float>(const int TransA, const int M, const int N, 
		const float alpha, const float* A, const float* x, const float beta, 
		float* y) {
		if (TransA == 0)
			JaffeGemm<float>(TransA, 0, M, 1, N, alpha, A, x, beta, y);
		else
			JaffeGemm<float>(TransA, 0, N, 1, M, alpha, A, x, beta, y);
	}

	template <>
	void JaffeGemv<double>(const int TransA, const int M, const int N, 
		const double alpha, const double* A, const double* x, const double beta, 
		double* y) {
		if (TransA == 0)
			JaffeGemm<double>(TransA, 0, M, 1, N, alpha, A, x, beta, y);
		else
			JaffeGemm<double>(TransA, 0, N, 1, M, alpha, A, x, beta, y);
	}

	template <typename Dtype>
	void JaffeCopy(const int N, const Dtype* X, Dtype* Y) {
		if (X != Y) {
			if (Jaffe::GetMode() == Jaffe::GPU) {
#ifndef CPU_ONLY
				CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
				NO_GPU;
#endif
			} else {
				memcpy(Y, X, sizeof(Dtype) * N);		
			}
		}
	}

	template void JaffeCopy<float>(const int N, const float* X, float* Y);
	template void JaffeCopy<double>(const int N, const double* X, double* Y);
	template void JaffeCopy<int>(const int N, const int* X, int* Y);
	template void JaffeCopy<unsigned int>(const int N, const unsigned int* X, 
		unsigned int* Y);

	template <typename Dtype>
	void JaffeSet(const int N, const Dtype alpha, Dtype* Y) {
		if (alpha == 0) {
			memset(Y, 0, sizeof(Dtype) * N);
			return;
		}
		for (int i = 0; i < N; i++) 
			Y[i] = alpha;
	}

	template void JaffeSet<float>(const int N, const float alpha, float* Y);
	template void JaffeSet<double>(const int N, const double alpha, double* Y);
	template void JaffeSet<int>(const int N, const int alpha, int* Y);

	template <>
	void JaffeAxpy<float>(const int N, const float alpha, const float* X, float* Y) {
		ArrayXf VX = Map<const ArrayXf>(X, N);
		ArrayXf VY = Map<ArrayXf>(Y, N);
		Map<ArrayXf>(Y, N) = alpha * VX + VY;
	}

	template <>
	void JaffeAxpy<double>(const int N, const double alpha, const double* X, double* Y) {
		ArrayXd VX = Map<const ArrayXd>(X, N);
		ArrayXd VY = Map<ArrayXd>(Y, N);
		Map<ArrayXd>(Y, N) = alpha * VX + VY;
	}

	template <>
	void JaffeSqr<float>(const int N, const float* a, float* y) {
		ArrayXf A = Map<const ArrayXf>(a, N);
		Map<ArrayXf>(y, N) = A.square();
	}

	template <>
	void JaffeSqr<double>(const int N, const double* a, double* y) {
		ArrayXd A = Map<const ArrayXd>(a, N);
		Map<ArrayXd>(y, N) = A.square();
	}

	template <>           
	void JaffePowx<float>(const int N, const float* a, const float b, float* y) {
		ArrayXf A = Map<const ArrayXf>(a, N);
		Map<ArrayXf>(y, N) = A.pow(b);
	}

	template <>           
	void JaffePowx<double>(const int N, const double* a, const double b, double* y) {
		ArrayXd A = Map<const ArrayXd>(a, N);
		Map<ArrayXd>(y, N) = A.pow(b);
	}

	// 向量相乘
	template <>
	void JaffeMul<float>(const int N, const float* a, const float* b, float* y) {
		ArrayXf A = Map<const ArrayXf>(a, N);
		ArrayXf B = Map<const ArrayXf>(b, N);
		Map<ArrayXf>(y, N) = A * B;
	}

	template <>
	void JaffeMul<double>(const int N, const double* a, const double* b, double* y) {
		ArrayXd A = Map<const ArrayXd>(a, N);
		ArrayXd B = Map<const ArrayXd>(b, N);
		Map<ArrayXd>(y, N) = A * B;
	}

	template <>
	void JaffeScal<float>(const int N, const float alpha, float* x) {
		ArrayXf X = Map<ArrayXf>(x, N);
		Map<ArrayXf>(x, N) = alpha * X;
	}

	template <>
	void JaffeScal<double>(const int N, const double alpha, double* x) {
		ArrayXd X = Map<ArrayXd>(x, N);
		Map<ArrayXd>(x, N) = alpha * X;
	}

	template <>
	void JaffeAddScalar<float>(const int N, const float alpha, float* y) {
		ArrayXf Y = Map<ArrayXf> (y, N);
		Map<ArrayXf> (y, N) = alpha + Y;
	}

	template <>
	void JaffeAddScalar<double>(const int N, const double alpha, double* y) {
		ArrayXd Y = Map<ArrayXd> (y, N);
		Map<ArrayXd> (y, N) = Y + alpha;
	}

	template <>
	void JaffeExp<float>(const int N, const float* a, float* y) {
		ArrayXf A = Map<const ArrayXf> (a, N);
		Map<ArrayXf> (y, N) = A.exp();
	}

	template <>
	void JaffeExp<double>(const int N, const double* a, double* y) {
		ArrayXd A = Map<const ArrayXd> (a, N);
		Map<ArrayXd> (y, N) = A.exp();
	}

	template <>
	void JaffeDiv<float>(const int N, const float* a, const float* b, float* y) {
		ArrayXf A = Map<const ArrayXf> (a, N);
		ArrayXf B = Map<const ArrayXf> (b, N);
		Map<ArrayXf>(y, N) = A / B;
	}

	template <>
	void JaffeDiv<double>(const int N, const double* a, const double* b, double* y) {
		ArrayXd A = Map<const ArrayXd> (a, N);
		ArrayXd B = Map<const ArrayXd> (b, N);
		Map<ArrayXd>(y, N) = A / B;
	}

	template <>
	float JaffeStridedDot<float>(const int N, const float* x, const int incx,
		const float* y, const int incy) {
		int size_x = 1 + (N - 1) * abs(incx);
		int size_y = 1 + (N - 1) * abs(incy);
		VectorXf X = Map<const VectorXf>(x, size_x);
		VectorXf Y = Map<const VectorXf>(y, size_y);
		return X.dot(Y);
	}

	template <>
	double JaffeStridedDot<double>(const int N, const double* x, const int incx,
		const double* y, const int incy) {
		int size_x = 1 + (N - 1) * abs(incx);
		int size_y = 1 + (N - 1) * abs(incy);
		VectorXd X = Map<const VectorXd>(x, size_x);
		VectorXd Y = Map<const VectorXd>(y, size_y);
		return X.dot(Y);
	}

	template <typename Dtype>
	Dtype JaffeDot(const int N, const Dtype* x, const Dtype* y) {
		return JaffeStridedDot(N, x, 1, y, 1);
	}

	template float JaffeDot(const int N, const float* x, const float* y);	
	template double JaffeDot(const int N, const double* x, const double* y);

} // namespace jaffe
