//
// Created by bluuuuer on 16-4-10.
//

#include "util/im2col.h"

namespace jaffe {

    template <typename Dtype>
    inline void Im2colNdCore(const Dtype* data_input, const bool im2col, 
		const int num_spatial_axes, const int* im_shape, const int* col_shape,
       	const int* kernel_shape, const int* pad, const int* stride, 
		const int* dilation, Dtype* data_output) {
        if (!im2col){
			int im_size = im_shape[0];
			for (int i = 0; i < num_spatial_axes; i++) {
				im_size *= im_shape[1 + i];
			}
			JaffeSet(im_size, Dtype(0), data_output);
        }
        int kernel_size = 1;
        for(int i = 0; i < num_spatial_axes; i++) {
            kernel_size *= kernel_shape[i];
        }
        const int channels_col = col_shape[0];
        vector<int> d_offset(num_spatial_axes, 0);
        vector<int> d_iter(num_spatial_axes, 0);
        for (int c_col = 0; c_col < channels_col; c_col++) {
            int offset = c_col;
            for(int d_i = num_spatial_axes - 1; d_i >= 0; d_i--) {
                if(d_i < num_spatial_axes - 1)
                    offset /= kernel_shape[d_i + 1];
                d_offset[d_i] = offset % kernel_shape[d_i];
            }
            for (bool incremented = true; incremented; ) {
                int index_col = c_col;
                int index_im = c_col / kernel_size;
                bool is_padding = false;
                for (int d_i = 0; d_i < num_spatial_axes; d_i++) {
                    const int d = d_iter[d_i];
                    const int d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i] * 
						dilation[d_i];
                    is_padding |= d_im < 0 ||d_im >= im_shape[d_i + 1];
                    index_col *= col_shape[d_i + 1];
                    index_col += d;
                    index_im *= im_shape[d_i + 1];
                    index_im += d_im;
                }
                if (im2col){
                    if (is_padding)
                        data_output[index_col] = 0;
                    else
                        data_output[index_col] = data_input[index_im];
                }
                else if (!is_padding)
                    data_output[index_im] += data_input[index_col];

                incremented = false;
                for (int d_i = num_spatial_axes - 1; d_i >= 0; d_i--){
                    const int d_max = col_shape[d_i + 1];
                    if (d_iter[d_i] == d_max - 1)
                        d_iter[d_i] = 0;
					else {
						d_iter[d_i] ++;
                        incremented = true;
                        break;
                    }
                }
            } // while(incremented){
        } // for (int c = 0; c < channels_col; c++){
		std::cout << "Im2ColNdCore() Done" << std::endl;
    }

	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
		return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}

    template <typename Dtype>
    void Im2col(const Dtype* data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
				Dtype* data_col){
		//std::cout << "im2col.cpp->Im2col()" << std::endl;
        const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1))
			/ stride_h + 1;
        const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1))
			/ stride_w + 1;
        const int channels_size = height * width;
		/*std::cout << "height = " << height << std::endl;
		std::cout << "width = " << width << std::endl;
		std::cout << "pad_h = " << pad_h << std::endl;
		std::cout << "pad_w = " << pad_w << std::endl;
		std::cout << "dilation_h = " << dilation_h << std::endl;
		std::cout << "dilation_w = " << dilation_w << std::endl;
		std::cout << "kernel_h = " << kernel_h << std::endl;
		std::cout << "kernel_w = " << kernel_w << std::endl;
		std::cout << "stride_h = " << stride_h << std::endl;
		std::cout << "stride_w = " << stride_w << std::endl;*/

		for (int channel = channels; channel--; data_im += channels_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height))
							for (int output_cols = output_w; output_cols; 
								output_cols--)
								*(data_col++) = 0;
						else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; 
								output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width))
									*(data_col++) = data_im[input_row * width + 
										input_col];
								else
									*(data_col++) = 0;
								input_col += stride_w;
							}
						}
						input_row += stride_h;
                	}
            	}
			}
		}
		//std::cout << "Done" << std::endl;
    }

    template void Im2col<float>(const float* data_im, const int channels,
                                const int height, const int width,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w,
                                const int stride_h, const int stride_w,
								const int dilation_h, const int dilation_w,
                                float* data_col);

    template void Im2col<double>(const double* data_im, const int channels,
                                 const int height, const int width,
                                 const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w,
                                 const int stride_h, const int stride_w,
								 const int dilation_h, const int dilation_w,
                                 double* data_col);

    template <typename Dtype>
    void Im2colNd(const Dtype* data_im, const int num_spatial_axes,
                  const int* im_shape, const int* col_shape,
                  const int* kernel_shape, const int* pad,
                  const int* stride, const int* dilation, Dtype* data_col) {
        Im2colNdCore(data_im, true, num_spatial_axes, im_shape,col_shape,
            kernel_shape, pad, stride, dilation, data_col);
    }

    template void Im2colNd<float>(const float* data_im, const int num_spatial_axes,
                                  const int* im_shape, const int* col_shape,
                                  const int* kernel_shape, const int* pad,
                                  const int* stride, const int* dilation, 
								  float* data_col);
    template void Im2colNd<double>(const double* data_im, const int num_spatial_axes,
                                   const int* im_shape, const int* col_shape,
                                   const int* kernel_shape, const int* pad,
                                   const int* stride, const int* dilation,
								   double* data_col);

	template <typename Dtype>
	void Col2im(const Dtype* data_col, const int channels, const int height, 
		const int width, const int kernel_h, const int kernel_w, const int pad_h,
		const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
		const int dilation_w, Dtype* data_im) {

			/*
		cout << "channels = " << channels << endl;
		cout << "height = " << height << endl;
		cout << "width = " << width << endl;
		cout << "kernel_h = " << kernel_h << endl;
		cout << "kernel_w = " << kernel_w << endl;
		cout << "pad_h = " << pad_h << endl;
		cout << "pad_w = " << pad_w << endl;
		cout << "stride_h = " << stride_h << endl;
		cout << "stride_w = " << stride_w << endl;
		cout << "dilation_h = " << dilation_h << endl;
		cout << "dilation_w = " << dilation_w << endl; 
		*/

		JaffeSet(height * width * channels, Dtype(0), data_im);
		const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1))
			/ stride_h + 1;
		const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1))
			/ stride_w + 1;
		const int channel_size = height * width;
		for (int channel = channels; channel --; data_im += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row ++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col ++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							data_col += output_w;
						} else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									int index = input_row * width + input_col;
									data_im[index] += *data_col;
								}
								data_col ++;
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}
	template void Col2im<float>(const float* data_col, const int channels, 
		const int height, const int width, const int kernel_h, const int kernel_w, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
		const int dilation_h, const int dilation_w, float* data_im);
	template void Col2im<double>(const double* data_col, const int channels, 
		const int height, const int width, const int kernel_h, const int kernel_w, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
		const int dilation_h, const int dilation_w, double* data_im);

	template <typename Dtype>
	void Col2imNd(const Dtype* data_col, const int num_spatial_axes,
			const int* im_shape, const int* col_shape, const int* kernel_shape, 
			const int* pad, const int* stride, const int* dilation, Dtype* data_im) {
		std::cout << "Col2imNd()" << std::endl;
		Im2colNdCore(data_col, false, num_spatial_axes, im_shape, col_shape, 
			kernel_shape, pad, stride, dilation, data_im);
		std::cout << "Done Col2imNd()" << std::endl;
	}
	template void Col2imNd<float>(const float* data_col, const int num_spatial_axes,
		const int* im_shape, const int* col_shape, const int* kernel_shape, 
		const int* pad, const int* stride, const int* dilation, float* data_im);
	template void Col2imNd<double>(const double* data_col, const int num_spatial_axes,
		const int* im_shape, const int* col_shape, const int* kernel_shape, 
		const int* pad, const int* stride, const int* dilation, double* data_im);

} // namespace jaffe
