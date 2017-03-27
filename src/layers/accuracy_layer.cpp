#include "layers/accuracy_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JAccuracyLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		m_top_k = this->m_layer_param.accuracy_param().top_k();
		m_has_ignore_label = 
			this->m_layer_param.accuracy_param().has_ignore_label();
		if (m_has_ignore_label)
			m_ignore_label = this->m_layer_param.accuracy_param().ignore_label();
	}

	template <typename Dtype>
	void JAccuracyLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		//cout << "JAccuracyLayer::Reshape()" << endl;
		m_label_axis = 
			bottom[0]->CanonicalAxisIndex(this->m_layer_param.accuracy_param().axis());
		m_outer_num = bottom[0]->GetCount(0, m_label_axis);
		m_inner_num = bottom[0]->GetCount(m_label_axis + 1);
		vector<int> top_shape(0);
		top[0]->Reshape(top_shape);
		if (top.size() > 1) {
			vector<int> top_shape_per_class(1);
			top_shape_per_class[0] = bottom[0]->GetShape(m_label_axis);
			top[1]->Reshape(top_shape_per_class);
			m_nums_buffer.Reshape(top_shape_per_class);
		}
	}

	template <typename Dtype>
	void JAccuracyLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
										const vector<JBlob<Dtype>*>& top) {
		//cout << "JAccuracyLayer::Forward()" << endl;
		Dtype accuracy = 0;
		const Dtype* bottom_data = bottom[0]->GetData();
		const Dtype* bottom_label = bottom[1]->GetData();
		const int dim = bottom[0]->GetCount() / m_outer_num;
		const int num_labels = bottom[0]->GetShape(m_label_axis);
		//vector<Dtype> maxval(m_top_k + 1);
		//vector<int> maxid(m_top_k + 1);
		if (top.size() > 1){
			JaffeSet(m_nums_buffer.GetCount(), Dtype(0), m_nums_buffer.GetMutableData());
			JaffeSet(top[1]->GetCount(), Dtype(0), top[1]->GetMutableData());
		}
		int count = 0;
		for (int i = 0; i < m_outer_num; i++)
			for (int j = 0; j < m_inner_num; j++) {
				const int label_value = 
					static_cast<int>(bottom_label[i * m_inner_num + j]);
				if (m_has_ignore_label && label_value == m_ignore_label)
					continue;
				if (top.size() > 1)
					m_nums_buffer.GetMutableData()[label_value] ++;
				// Top-k accuracy
				vector<pair<Dtype, int> > bottom_data_vector;
				for (int k = 0; k < num_labels; k ++)
					bottom_data_vector.push_back(make_pair(
						bottom_data[i * dim + k * m_inner_num + j], k));
				partial_sort(bottom_data_vector.begin(), 
					bottom_data_vector.begin() + m_top_k, bottom_data_vector.end(),
					greater<pair<Dtype, int> >());
				// check if true label is in top k predictions
				for (int k = 0; k < m_top_k; k++)
					if (bottom_data_vector[k].second == label_value) {
						accuracy++;
						if (top.size() > 1)
							top[1]->GetMutableData()[label_value]++;
						break;
					}
				count++;
			}
		top[0]->GetMutableData()[0] = accuracy / count;
		if (top.size() > 1)
			for (int i = 0; i < top[1]->GetCount(); i++)
				top[1]->GetMutableData()[i] = m_nums_buffer.GetData()[i] == 0 ? 0 :
					top[1]->GetData()[i] / m_nums_buffer.GetData()[i];
	}

	template class JAccuracyLayer <double>;
	template class JAccuracyLayer <float>;

	REGISTER_LAYER_CLASS(Accuracy);

} // namespace jaffe
