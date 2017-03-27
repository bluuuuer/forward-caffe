#include "layers/softmax_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JSoftmaxLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		m_softmax_axis = 
			bottom[0]->CanonicalAxisIndex(this->m_layer_param.softmax_param().axis());
		top[0]->ReshapeLike(*bottom[0]);
		vector<int> mult_dims(1, bottom[0]->GetShape(m_softmax_axis));
		m_sum_multiplier.Reshape(mult_dims);
		Dtype* multiplier_data = m_sum_multiplier.GetMutableData();
		JaffeSet(m_sum_multiplier.GetCount(), Dtype(1), multiplier_data);
		m_outer_num = bottom[0]->GetCount(0, m_softmax_axis);
		m_inner_num = bottom[0]->GetCount(m_softmax_axis + 1);
		vector<int> scale_dims = bottom[0]->GetShape();
		scale_dims[m_softmax_axis] = 1;
		m_scale.Reshape(scale_dims);
	}

	template <typename Dtype>
	void JSoftmaxLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();
		Dtype* scale_data = m_scale.GetMutableData();

		int channels = bottom[0]->GetShape(m_softmax_axis);
		int dim = bottom[0]->GetCount() / m_outer_num;
		JaffeCopy(bottom[0]->GetCount(), bottom_data, top_data);
		// 先取最大，再计算e指数，再归一化
		for (int i = 0; i < m_outer_num; i++) {
			// 初始化 scale_data
			JaffeCopy(m_inner_num, bottom_data + i * dim, scale_data);
			for (int j = 0; j < channels; j++)
				for (int k = 0; k < m_inner_num; k++)
					scale_data[k] = max(scale_data[k], 
						bottom_data[i * dim + j * m_inner_num + k]);
			// 做减法
			JaffeGemm<Dtype>(0, 0, channels, m_inner_num, 1, -1., 
				m_sum_multiplier.GetData(), scale_data, 1., top_data);
			// 求幂
			JaffeExp<Dtype>(dim, top_data, top_data);
			// 求 e 指数后求和
			JaffeGemv<Dtype>(1, channels, m_inner_num, 1., top_data,
				m_sum_multiplier.GetData(), 0., scale_data);
			// 求余
			for (int j = 0; j < channels; j++) {
				JaffeDiv(m_inner_num, top_data, scale_data, top_data);
				top_data += m_inner_num;
			}
		}
	}

	template class JSoftmaxLayer <double>;
	template class JSoftmaxLayer <float>;
#ifdef CPU_ONLY
	STUB_GPU(JSoftmaxLayer);
#endif

	REGISTER_LAYER_CLASS(Softmax);

} // namespace jaffe
