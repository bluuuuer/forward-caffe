#include "layers/input_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JInputLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const int num_top = top.size();
		const InputParameter& ip_param = this->m_layer_param.input_param();
		const int num_shape = ip_param.shape_size();
		if (num_shape > 0) 
			for (int i = 0; i < num_top; i++) {
				const int shape_index = (ip_param.shape_size() == 1) ? 0 : i;
				top[i]->Reshape(ip_param.shape(shape_index));
			}
	}

	template class JInputLayer <float>;
	template class JInputLayer <double>;

	REGISTER_LAYER_CLASS(Input);
	
} // namespace jaffe
