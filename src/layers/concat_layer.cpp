#include "layers/concat_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JConcatLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
  		const ConcatParameter& concat_param = this->m_layer_param.concat_param();
		if (concat_param.has_axis() && concat_param.has_concat_dim()) {
			cout << "Either axis or concat_dim should be specified; not both."
				<< endl;
		}
	}

	template <typename Dtype>
	void JConcatLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		const int num_axes = bottom[0]->GetNumAxes();
		const ConcatParameter& concat_param = this->m_layer_param.concat_param();
  		if (concat_param.has_concat_dim()) {
    		m_concat_axis = static_cast<int>(concat_param.concat_dim());
    		// Don't allow negative indexing for concat_dim, a uint32 -- almost
    		// certainly unintended.
  		} else {
    		m_concat_axis = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  		}
  		// Initialize with the first blob.
  		vector<int> top_shape = bottom[0]->GetShape();
  		m_num_concats = bottom[0]->GetCount(0, m_concat_axis);
  		m_concat_input_size = bottom[0]->GetCount(m_concat_axis + 1);
  		for (int i = 1; i < bottom.size(); ++i) {
			if (!num_axes == bottom[i]->GetNumAxes()) {
        		cout <<  "All inputs must have the same #axes." << endl;
			}
 	   		for (int j = 0; j < num_axes; ++j) {
      			if (j == m_concat_axis) { 
					continue; 
				}
				if (!top_shape[j] == bottom[i]->GetShape(j)) {
          			cout << "All inputs must have the same shape, except at " <<
						"concat_axis." << endl;
				}
			}
    		top_shape[m_concat_axis] += bottom[i]->GetShape(m_concat_axis);
  		}
  		top[0]->Reshape(top_shape);
  		if (bottom.size() == 1) {
    		top[0]->ShareData(*bottom[0]);
    		top[0]->ShareDiff(*bottom[0]);
  		}
	}

	template <typename Dtype>
	void JConcatLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
      	const vector<JBlob<Dtype>*>& top) {
  		if (bottom.size() == 1) { 
			return; 
		}
  		Dtype* top_data = top[0]->GetMutableData();
  		int offset_axis = 0;
  		const int top_axis = top[0]->GetShape(m_concat_axis);
  		for (int i = 0; i < bottom.size(); ++i) {
    		const Dtype* bottom_data = bottom[i]->GetData();
    		const int bottom_axis = bottom[i]->GetShape(m_concat_axis);
    		for (int n = 0; n < m_num_concats; ++n) {
      			JaffeCopy(bottom_axis * m_concat_input_size,
          			bottom_data + n * bottom_axis * m_concat_input_size,
          			top_data + (n * top_axis + offset_axis) * m_concat_input_size);
    		}
    		offset_axis += bottom_axis;
  		}
	}

	template class JConcatLayer<float>;
	template class JConcatLayer<double>;

#ifdef CPU_ONLY
	STUB_GPU(JConcatLayer);
#endif

	REGISTER_LAYER_CLASS(Concat);

}  // namespace jaffe
