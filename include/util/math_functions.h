// huangshize 2016.03.16
// === math_function.h ===
// 对应原来的caffe/include/caffe/util/math_function.hpp文件
// 提供了一系列的数学工具支持

#ifndef JAFFE_MATH_FUNCTIONS_H_
#define JAFFE_MATH_FUNCTIONS_H_

#include <string>


// 用于对指针所指定的有限空间进行“赋值”
inline void jaffe_memset(const size_t N, const int alpha, void* X) {
	//memset(X, alpha, N);  // 在头文件<string>内
}

template <typename Dtype>
Dtype jaffe_asum(const int n, const Dtype* x);

template <typename Dtype>
Dtype jaffe_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
void jaffe_axpy(const int n, const Dtype alpha, const Dtype* x, const Dtype* y);

template <typename Dtype>
void jaffe_scale(const int n, const Dtype alpha, const Dtype * x);

#endif