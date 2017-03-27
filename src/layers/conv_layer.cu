#include "layers/conv_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JConvolutionLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
      		const vector<JBlob<Dtype>*>& top) {
			
  		const Dtype* weight = this->m_blobs[0]->GetGpuData();
  		for (int i = 0; i < bottom.size(); ++i) {
    		const Dtype* bottom_data = bottom[i]->GetGpuData();
    		Dtype* top_data = top[i]->GetMutableGpuData();
    		for (int n = 0; n < this->m_num; ++n) {
      			this->ForwardGpuGemm(bottom_data + n * this->m_bottom_dim, weight,
          			top_data + n * this->m_top_dim);
      			if (this->m_bias_term) {
       	 			const Dtype* bias = this->m_blobs[1]->GetGpuData();
        			this->ForwardGpuBias(top_data + n * this->m_top_dim, bias);
      			}
    		}
  		}
	}

	INSTANTIATE_LAYER_GPU_FORWARD(JConvolutionLayer);
}  // namespace jaffe
