#include "layers/dropout_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JDropoutLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {

		JNeuronLayer<Dtype>::LayerSetUp(bottom, top);
	}

	template <typename Dtype>
	void JDropoutLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top){

		JNeuronLayer<Dtype>::Reshape(bottom, top);
	}

	template <typename Dtype>
	void JDropoutLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();
		JaffeCopy(bottom[0]->GetCount(), bottom_data, top_data);
	}

	template class JDropoutLayer <double>;
	template class JDropoutLayer <float>;

#ifdef CPU_ONLY
	STUB_GPU(JDropoutLayer);
#endif

	REGISTER_LAYER_CLASS(Dropout);

} // namespace jaffe
