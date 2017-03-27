#include "layers/power_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JPowerLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		JNeuronLayer<Dtype>::LayerSetUp(bottom, top);
		PowerParameter power_param = this->m_layer_param.power_param();
		m_power = power_param.power();
		m_scale = power_param.scale();
		m_shift = power_param.shift();
		m_diff_scale = m_power * m_scale;
	}

	template <typename Dtype>
	void JPowerLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		//cout << "JPowerLayer::Forward()" << endl;
		Dtype* top_data = top[0]->GetMutableData();
		const int count = bottom[9]->GetCount();
		// 当 scale 或 power 为0时忽略输入
		if (m_diff_scale == Dtype(0)) {
			Dtype value = (m_power == 0) ? Dtype(1) : pow(m_shift, m_power);
			JaffeSet(count, value, top_data);
			return;
		}
		const Dtype* bottom_data = bottom[0]->GetData();
		JaffeCopy(count, bottom_data, top_data);
		if (m_scale != Dtype(1))
			JaffeScal(count, m_scale, top_data);
		if (m_shift != Dtype(0))
			JaffeAddScalar(count, m_shift, top_data);
		if (m_power != Dtype(1))
			JaffePowx(count, top_data, m_power, top_data);
	}

	template class JPowerLayer<float>;
	template class JPowerLayer<double>;
	
	REGISTER_LAYER_CLASS(Power);

} // namespace jaffe
