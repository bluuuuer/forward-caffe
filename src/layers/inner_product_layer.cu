#include "layers/inner_product_layer.h"

namespace jaffe {
		
	template <typename Dtype>
	void JInnerProductLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->GetGpuData();
		Dtype* top_data = top[0]->GetMutableGpuData();
		const Dtype* weight = this->m_blobs[0]->GetGpuData();
		if (m_M == 1) {
			JaffeGpuGemv<Dtype>(0, m_N, m_K, (Dtype)1., weight, bottom_data,
				(Dtype)0., top_data);
			if (m_bias_term) {
				JaffeGpuAxpy<Dtype>(m_N, m_bias_multiplier.GetData()[0],
					this->m_blobs[1]->GetGpuData(), top_data);
			}
		} else {
			JaffeGpuGemm<Dtype>(0, m_transpose ? 0 : 1, m_M, m_N, m_K, (Dtype)1.,
				bottom_data, weight, (Dtype)0., top_data);
			if (m_bias_term) {
				const Dtype* m_bias_multiplier_temp = m_bias_multiplier.GetGpuData();
				const Dtype* m_blobs_temp = this->m_blobs[1]->GetGpuData();
				JaffeGpuGemm<Dtype>(0, 0, m_M, m_N, 1, (Dtype)1.,
					m_bias_multiplier_temp, m_blobs_temp,
					(Dtype)1., top_data);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FORWARD(JInnerProductLayer);
	
} // namespace jaffe
