#include "layers/relu_layer.h"

namespace jaffe {

	template <typename Dtype>
	__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    	Dtype negative_slope) {
			
 		CUDA_KERNEL_LOOP(index, n) {
    		out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  		}	
	}

	template <typename Dtype>
	void JReLULayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
    	const vector<JBlob<Dtype>*>& top) {

  		const Dtype* bottom_data = bottom[0]->GetGpuData();
  		Dtype* top_data = top[0]->GetMutableGpuData();
  		const int count = bottom[0]->GetCount();
  		Dtype negative_slope = this->m_layer_param.relu_param().negative_slope();

  		ReLUForward<Dtype><<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(
      		count, bottom_data, top_data, negative_slope);
  		CUDA_POST_KERNEL_CHECK;
	}

INSTANTIATE_LAYER_GPU_FORWARD(JReLULayer);

}  // namespace jaffe
