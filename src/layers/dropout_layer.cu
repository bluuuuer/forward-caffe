#include "layers/dropout_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JDropoutLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetGpuData();
		Dtype* top_data = top[0]->GetMutableGpuData();
		const int count = bottom[0]->GetCount();
		JaffeCopy(count, bottom_data, top_data);
	}

	INSTANTIATE_LAYER_GPU_FORWARD(JDropoutLayer);
	
} // namespace jaffe
