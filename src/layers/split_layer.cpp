#include "layers/split_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JSplitLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
//		cout << "JSplitLayer::Reshape()" << endl;
		for (int i = 0; i < top.size(); i++)
			top[i]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void JSplitLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		//cout << "JSplitLayer::Forward()" << endl;
		for (int i = 0; i < top.size(); i++)
			top[i]->ShareData(*bottom[0]);
	}
	
	template class JSplitLayer<float>;
	template class JSplitLayer<double>;

	REGISTER_LAYER_CLASS(Split);

} // namespace jaffe
