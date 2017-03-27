//
// Created by bluuuuer on 16-4-10.
//

#ifndef JAFFE_IM2COL_H
#define JAFFE_IM2COL_H

#include <vector>
#include <iostream>

#include "util/math_functions.hpp"

using std::vector;
using std::cout;
using std::endl;

namespace jaffe {

    template <typename Dtype>
    void Im2colNd(const Dtype* data_im, const int num_spatial_axes,
                  const int* im_shape, const int* col_shape,
                  const int* kernel_shape, const int* pad,
                  const int* stride, const int* dilation, Dtype* data_col); // ԭ im2col_nd_cpu()

    template <typename Dtype>
    void Im2col(const Dtype* data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w, 
				Dtype* data_col); // ԭ im2col_cpu()

	template <typename Dtype>
	void Col2imNd(const Dtype* data_col, const int num_spatial_axes, 
		const int* im_shape, const int* col_shape, const int* kernel_shape, 
		const int* pad, const int* stride, const int* dilation, Dtype* data_im);

	template <typename Dtype>
	void Col2im(const Dtype* data_col, const int channels, const int height, 
		const int width, const int kernel_h, const int kernel_w, const int pad_h,
		const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
		const int dilation_w, Dtype* data_im);

	template <typename Dtype>
	void Im2colGpu(const Dtype* data_im, const int channels, const int height, 
		const int width, const int kernel_h, const int kernle_w, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
		const int dilation_h, const int dilation_w, Dtype* data_col);

	template <typename Dtype>
	void Im2colNdGpu(const Dtype* data_im, const int num_spatial_axes, 
		const int col_size, const int* im_shape, const int* col_shape, 
		const int* kernel_shape, const int* pad, const int* stride,
		const int* dilation, Dtype* data_col);

} // namespace jaffe
#endif //JAFFE_IM2COL_H
