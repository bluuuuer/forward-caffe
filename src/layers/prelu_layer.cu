#include "layers/prelu_layer.h"

namespace jaffe {

	// CUDA kernele for forward
	template <typename Dtype>
	__global__ void PReLUForward(const int n, const int channels, const int dim,
    	const Dtype* in, Dtype* out, const Dtype* slope_data, const int div_factor) {
			
  		CUDA_KERNEL_LOOP(index, n) {
    		int c = (index / dim) % channels / div_factor;
    		out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  		}
	}

	template <typename Dtype>
	void JPReLULayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
    	const vector<JBlob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->GetGpuData();
  		Dtype* top_data = top[0]->GetMutableGpuData();
  		const int count = bottom[0]->GetCount();
  		const int dim = bottom[0]->GetCount(2);
  		const int channels = bottom[0]->GetChannels();
  		const Dtype* slope_data = this->m_blobs[0]->GetGpuData();
  		const int div_factor = m_channel_shared? channels : 1;
	
  		// For in-place computation
  		if (top[0] == bottom[0]) {
    		JaffeCopy(count, bottom_data, m_bottom_memory.GetMutableGpuData());
  		}

  		PReLUForward<Dtype><<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(
      		count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  		CUDA_POST_KERNEL_CHECK;
	}

INSTANTIATE_LAYER_GPU_FORWARD(JPReLULayer);

}  // namespace jaffe
