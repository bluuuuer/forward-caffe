#include "layers/softmax_layer.h"

namespace jaffe {

	template <typename Dtype>
	__global__ void kernel_channel_max(const int num, const int channels,
    	const int spatial_dim, const Dtype* data, Dtype* out) {
			
  		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    		int n = index / spatial_dim;
    		int s = index % spatial_dim;
    		Dtype maxval = -FLT_MAX;
    		for (int c = 0; c < channels; ++c) {
      			maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    		}
    		out[index] = maxval;
  		}
	}

	template <typename Dtype>
	__global__ void kernel_channel_subtract(const int count, const int num, 
		const int channels, const int spatial_dim, const Dtype* channel_max, 
		Dtype* data) {
			
  		CUDA_KERNEL_LOOP(index, count) {
    		int n = index / channels / spatial_dim;
    		int s = index % spatial_dim;
    		data[index] -= channel_max[n * spatial_dim + s];
  		}
	}

	template <typename Dtype>
	__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  		CUDA_KERNEL_LOOP(index, count) {
    		out[index] = exp(data[index]);
  		}
	}

	template <typename Dtype>
	__global__ void kernel_channel_sum(const int num, const int channels,
    	const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    		int n = index / spatial_dim;
    		int s = index % spatial_dim;
    		Dtype sum = 0;
    		for (int c = 0; c < channels; ++c) {
      			sum += data[(n * channels + c) * spatial_dim + s];
    		}
    		channel_sum[index] = sum;
  		}	
	}

	template <typename Dtype>
	__global__ void kernel_channel_div(const int count,
    	const int num, const int channels,
    	const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  	
		CUDA_KERNEL_LOOP(index, count) {
    		int n = index / channels / spatial_dim;
    		int s = index % spatial_dim;
    		data[index] /= channel_sum[n * spatial_dim + s];
  		}
	}

	template <typename Dtype>
	__global__ void kernel_channel_dot(const int num, const int channels,
    	const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    	Dtype* channel_dot) {
  
		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    		int n = index / spatial_dim;
    		int s = index % spatial_dim;
    		Dtype dot = 0;
    		for (int c = 0; c < channels; ++c) {
    	  		dot += (data_1[(n * channels + c) * spatial_dim + s]
          			* data_2[(n * channels + c) * spatial_dim + s]);
    		}
    		channel_dot[index] = dot;
  		}
	}	

	template <typename Dtype>
	void JSoftmaxLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
    	const vector<JBlob<Dtype>*>& top) {
  
		const Dtype* bottom_data = bottom[0]->GetGpuData();
  		Dtype* top_data = top[0]->GetMutableGpuData();
  		Dtype* scale_data = m_scale.GetMutableGpuData();
  		int count = bottom[0]->GetCount();
  		int channels = top[0]->GetShape(m_softmax_axis);
  		JaffeCopy(count, bottom_data, top_data);
		
  // compute max
  		kernel_channel_max<Dtype><<<JAFFE_GET_BLOCKS(m_outer_num * m_inner_num),
      		JAFFE_CUDA_NUM_THREADS>>>(m_outer_num, channels, m_inner_num, top_data,
      		scale_data);
  // subtract
  		kernel_channel_subtract<Dtype><<<JAFFE_GET_BLOCKS(count),
      		JAFFE_CUDA_NUM_THREADS>>>(count, m_outer_num, channels, m_inner_num,
      		scale_data, top_data);
  // exponentiate
  		kernel_exp<Dtype><<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(
      		count, top_data, top_data);
  // sum after exp
  		kernel_channel_sum<Dtype><<<JAFFE_GET_BLOCKS(m_outer_num * m_inner_num),
      		JAFFE_CUDA_NUM_THREADS>>>(m_outer_num, channels, m_inner_num, top_data,
      		scale_data);
  // divide
  		kernel_channel_div<Dtype><<<JAFFE_GET_BLOCKS(count),
      		JAFFE_CUDA_NUM_THREADS>>>(count, m_outer_num, channels, m_inner_num,
      	scale_data, top_data);
	}

INSTANTIATE_LAYER_GPU_FORWARD(JSoftmaxLayer);

}  // namespace caffe
