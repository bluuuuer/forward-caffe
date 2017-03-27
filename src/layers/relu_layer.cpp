#include "layers/relu_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JReLULayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();

		const int count = bottom[0]->GetCount();
		Dtype negative_slope = this->m_layer_param.relu_param().negative_slope();
		for (int i = 0; i < count; i++) {
			//------不同于标准的 ReLU 函数的 max(x,0)，当 x>0 时 输出 x，当 x<=0
			//------时输出 negative_slope * min(x,0)。negative_slope 在 prototxt 中
			//------由用户定义大小，默认值为0。
			top_data[i] = std::max(bottom_data[i], Dtype(0)) + 
				negative_slope * std::min(bottom_data[i], Dtype(0));
		}
	}

	template class JReLULayer <double>;
	template class JReLULayer <float>;

#ifdef CPU_ONLY
	STUB_GPU(JReLULayer);
#endif

	REGISTER_LAYER_CLASS(ReLU);

} // namespace jaffe
