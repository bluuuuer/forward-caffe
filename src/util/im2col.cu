#include "util/im2col.h"

namespace jaffe {

	template <typename Dtype>
	__global__ void Im2colGpuKernel(const int n, const Dtype* data_im, 
		const int height, const int width, const int kernel_h, const int kernel_w,
    	const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    	const int dilation_h, const int dilation_w, const int height_col, 
		const int width_col, Dtype* data_col) {
  		CUDA_KERNEL_LOOP(index, n) {
    		const int h_index = index / width_col;
    		const int h_col = h_index % height_col;
    		const int w_col = index % width_col;
    		const int c_im = h_index / height_col;
    		const int c_col = c_im * kernel_h * kernel_w;
    		const int h_offset = h_col * stride_h - pad_h;
    		const int w_offset = w_col * stride_w - pad_w;
    		Dtype* data_col_ptr = data_col;
    		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    		const Dtype* data_im_ptr = data_im;
    		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    		for (int i = 0; i < kernel_h; ++i) {
      			for (int j = 0; j < kernel_w; ++j) {
        			int h_im = h_offset + i * dilation_h;
        			int w_im = w_offset + j * dilation_w;
        			*data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && 
						w_im < width) ? data_im_ptr[i * dilation_h * width + 
						j * dilation_w] : 0;
        			data_col_ptr += height_col * width_col;
      			}
    		}
  		}
	}
	
	template <typename Dtype>
	void Im2colGpu(const Dtype* data_im, const int channels, 
		const int height, const int width, const int kernel_h, const int kernel_w, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w, Dtype* data_col) {
  		int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / 
			stride_h + 1;
  		int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / 
			stride_w + 1;
  		int num_kernels = channels * height_col * width_col;
  		Im2colGpuKernel<Dtype><<<JAFFE_GET_BLOCKS(num_kernels), 
			JAFFE_CUDA_NUM_THREADS>>>( num_kernels, data_im, height, width, kernel_h, 
			kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
			height_col, width_col, data_col);
  		CUDA_POST_KERNEL_CHECK;
	}
	template void Im2colGpu<float>(const float* data_im, const int channels, 
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
		const int dilation_h, const int dilation_w, float* data_col);
	template void Im2colGpu<double>(const double* data_im, const int channels, 
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
		const int dilation_h, const int dilation_w, double* data_col);

	template <typename Dtype, int num_axes>
	__global__ void Im2colNdGpuKernel(const int n, const Dtype* data_im,
    	const int* im_shape, const int* col_shape, const int* kernel_shape, 
		const int* pad, const int* stride, const int* dilation, Dtype* data_col) {
  		int d_temp[num_axes];  // NOLINT(runtime/arrays)
  		int d_iter[num_axes];  // NOLINT(runtime/arrays)

  		__shared__ int shared_dilation[num_axes];
  		__shared__ int shared_kernel_shape[num_axes];
  		__shared__ int shared_pad[num_axes];
  		__shared__ int shared_stride[num_axes];
  		__shared__ int shared_col_shape[num_axes + 1];
  		__shared__ int shared_im_shape[num_axes + 1];

  		if (threadIdx.x < num_axes) {
    		shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    		shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    		shared_pad[threadIdx.x] = pad[threadIdx.x];
    		shared_stride[threadIdx.x] = stride[threadIdx.x];
  		}
  		if (threadIdx.x < num_axes + 1) {
   	 		shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    		shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  		}
  		__syncthreads();

  		int i;
  		CUDA_KERNEL_LOOP(index, n) {
    		int channel_in = index;
    		int channel_out = 1;
    		for (i = num_axes - 1; i >= 0; --i) {
      			d_temp[i] = channel_in % shared_col_shape[i + 1];
      			channel_in /= shared_col_shape[i + 1];
      			channel_out *= shared_kernel_shape[i];
    		}
    		channel_out *= channel_in;
    		int data_col_inc = 1;
    		for (i = 0; i < num_axes; ++i) {
      			channel_out *= shared_col_shape[i + 1];
      			channel_out += d_temp[i];
      			d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      			channel_in *= shared_im_shape[i + 1];
      			channel_in += d_temp[i];
      			data_col_inc *= shared_col_shape[i + 1];
      			d_iter[i] = 0;
    		}
    		Dtype* data_col_ptr = data_col + channel_out;
    		const Dtype* data_im_ptr = data_im + channel_in;
    		bool incremented;
    		do {
      			bool in_range = true;
      			for (i = 0; i < num_axes; ++i) {
        			const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        			in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        			if (!in_range) { 
						break; 
					}
      			}
      			if (in_range) {
        			int data_im_offset = d_iter[0] * shared_dilation[0];
        			for (i = 1; i < num_axes; ++i) {
          				data_im_offset *= shared_im_shape[i + 1];
          				data_im_offset += d_iter[i] * shared_dilation[i];
        			}
        			*data_col_ptr = data_im_ptr[data_im_offset];
      			} else {
        			*data_col_ptr = 0;
      			}
      			data_col_ptr += data_col_inc;
      			incremented = false;
      			for (i = num_axes - 1; i >= 0; --i) {
        			const int d_max = shared_kernel_shape[i];
        			if (d_iter[i] == d_max - 1) {
          				d_iter[i] = 0;
        			} else {  // d_iter[i] < d_max - 1
          				++d_iter[i];
          				incremented = true;
          				break;
        			}
      			}  // for (int i = num_axes - 1; i >= 0; --i)
    		} while (incremented);  // do
  		}  // CUDA_KERNEL_LOOP(index, n)
	}

	template <typename Dtype>
	void Im2colNdGpu(const Dtype* data_im, const int num_spatial_axes, 
		const int num_kernels, const int* im_shape, const int* col_shape, 
		const int* kernel_shape, const int* pad, const int* stride, 
		const int* dilation, Dtype* data_col) {
  // num_axes should be smaller than block size
  		switch (num_spatial_axes) {
  		case 1:
    		Im2colNdGpuKernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 2:
    		Im2colNdGpuKernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 3:
    		Im2colNdGpuKernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 4:
    		Im2colNdGpuKernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 5:
    		Im2colNdGpuKernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 6:
    		Im2colNdGpuKernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 7:
    		Im2colNdGpuKernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 8:
    		Im2colNdGpuKernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 9:
    		Im2colNdGpuKernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		case 10:
    		Im2colNdGpuKernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
          		<<<JAFFE_GET_BLOCKS(num_kernels), JAFFE_CUDA_NUM_THREADS>>>( 
				num_kernels, data_im, im_shape, col_shape, kernel_shape, pad, stride, 
				dilation, data_col);
    		break;
  		default:
			std::cout << __FILE__ << " line: " << __LINE__ << std::endl <<  
				"ERROR: Im2colNdGpu does not support computiation with " << 
				num_spatial_axes << " spatial axes" << endl;
  		}
  		CUDA_POST_KERNEL_CHECK;
	}
	template void Im2colNdGpu<float>(const float* data_col, 
		const int num_spatial_axes, const int im_size, const int* im_shape, 
		const int* col_shape, const int* kernel_shape, const int* pad, 
		const int* stride, const int* dilation, float* data_im);
	template void Im2colNdGpu<double>(const double* data_col,
    	const int num_spatial_axes, const int im_size, const int* im_shape, 
		const int* col_shape, const int* kernel_shape, const int* pad, 
		const int* stride, const int* dilation, double* data_im);
	
} // namespace jaffe
