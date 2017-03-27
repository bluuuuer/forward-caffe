#include "layers/loss_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JLossLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		if (this->m_layer_param.loss_weight_size() == 0)
			this->m_layer_param.add_loss_weight(Dtype(1));
	}

	template <typename Dtype>
	void JLossLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		cout << "JLossLayer::Reshape()" << endl;
		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);
	}

	template class JLossLayer <float>;
	template class JLossLayer <double>;

} // namespace jaffe
