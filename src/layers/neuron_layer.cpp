#include "layers/neuron_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JNeuronLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
//		cout << "JNeuronLayer::Reshape()" << endl;
		top[0]->ReshapeLike(*bottom[0]);
	}

	template class JNeuronLayer <float>;
	template class JNeuronLayer <double>;
} // namespace jaffe
