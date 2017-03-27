// huangshize 2016.03.16
// === math_function.h ===
// ��Ӧԭ����caffe/include/caffe/util/math_function.hpp�ļ�
// �ṩ��һϵ�е���ѧ����֧��

#ifndef JAFFE_MATH_FUNCTIONS_H_
#define JAFFE_MATH_FUNCTIONS_H_

#include <string>


// ���ڶ�ָ����ָ�������޿ռ���С���ֵ��
inline void jaffe_memset(const size_t N, const int alpha, void* X) {
	//memset(X, alpha, N);  // ��ͷ�ļ�<string>��
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