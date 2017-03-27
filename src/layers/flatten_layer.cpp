#include "layers/flatten_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JFlattenLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {
		FlattenParameter f_param = this->m_layer_param.flatten_param();
		const int start_axis = bottom[0]->CanonicalAxisIndex(f_param.axis());
		const int end_axis = bottom[0]->CanonicalAxisIndex(f_param.end_axis());
		vector<int> top_shape;
		for (int i = 0; i < start_axis; i++) {
			top_shape.push_back(bottom[0]->GetShape(i));
		}
		const int flattened_dim = bottom[0]->GetCount(start_axis, end_axis + 1);
		top_shape.push_back(flattened_dim);
		for (int i = end_axis + 1; i < bottom[0]->GetNumAxes(); i++) {
			top_shape.push_back(bottom[0]->GetShape(i));
		}
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void JFlattenLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {
		top[0]->ShareData(*bottom[0]);
	}

	template class JFlattenLayer<float>;
	template class JFlattenLayer<double>;
		
	REGISTER_LAYER_CLASS(Flatten);
} // namespace jaffe
