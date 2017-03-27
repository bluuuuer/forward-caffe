#include "layers/l2norm_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JL2NormLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->GetGpuData();
		Dtype* top_data = top[0]->GetMutableGpuData();
		const int num = bottom[0]->GetNum();
		const int all_fea_count = bottom[0]->GetCount();
		const int dim = all_fea_count / num;

		JaffeGpuMul(all_fea_count, bottom_data, bottom_data,
			m_temp.GetMutableGpuData());

		JaffeGpuGemv<Dtype>(0, num, dim, 1, m_temp.GetGpuData(), 
			m_sum_multiplier.GetGpuData(), 0., m_norm.GetMutableGpuData());

		JaffeGpuPowx(num, m_norm.GetGpuData(), Dtype(0.5), 
			m_norm.GetMutableGpuData());

		JaffeGpuAddScalar<Dtype>(num, m_eps, m_norm.GetMutableGpuData());

		JaffeGpuGemm<Dtype>(0, 0, num, dim, 1, 1, m_norm.GetGpuData(), 
			m_sum_multiplier.GetGpuData(), 0., m_temp.GetMutableGpuData());

		JaffeGpuDiv<Dtype>(all_fea_count, bottom_data, m_temp.GetGpuData(), top_data);
	
		CUDA_POST_KERNEL_CHECK;
	}

INSTANTIATE_LAYER_GPU_FORWARD(JL2NormLayer);

} // namespace jaffe
