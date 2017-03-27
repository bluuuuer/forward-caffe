#include "layers/eltwise_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JEltwiseLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		EltwiseParameter eltwise_param = this->m_layer_param.eltwise_param();
		m_op = eltwise_param.operation();
		m_coeffs = vector<Dtype>(bottom.size(), 1);
		if (eltwise_param.coeff_size())
			for (int i = 0; i < bottom.size(); i++)
				m_coeffs[i] = eltwise_param.coeff(i);
	}

	template <typename Dtype>
	void JEltwiseLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		//cout << "JEltwiseLayer::Reshape()" << endl;
		top[0]->ReshapeLike(*bottom[0]);
		if (m_op == EltwiseParameter_EltwiseOp_MAX && top.size() == 1)
			m_max_idx.Reshape(bottom[0]->GetShape());
	}

	template <typename Dtype>
	void JEltwiseLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		//cout << "JEltwiseLayer::Forward()" << endl;
		int* mask = NULL;
		const Dtype* bottom_data_a = NULL;
		const Dtype* bottom_data_b = NULL;
		const int count = top[0]->GetCount();
		Dtype* top_data = top[0]->GetMutableData();
		
		switch (m_op) {
		case EltwiseParameter_EltwiseOp_PROD:
			JaffeMul(count, bottom[0]->GetData(), bottom[1]->GetData(), top_data);
			for (int i = 0; i < bottom.size(); i++)
				JaffeMul(count, top_data, bottom[i]->GetData(), top_data);
			break;
		case EltwiseParameter_EltwiseOp_SUM:
			JaffeSet(count, Dtype(0), top_data);
			for (int i = 0; i < bottom.size(); i++) 
				JaffeAxpy(count, m_coeffs[i], bottom[i]->GetData(), top_data);
			break;
		case EltwiseParameter_EltwiseOp_MAX:
			// 初始化
			mask = m_max_idx.GetMutableData();
			JaffeSet(count, -1, mask);
			JaffeSet(count, Dtype(-FLT_MAX), top_data);
			// bottom 0 & 1
			bottom_data_a = bottom[0]->GetData();
			bottom_data_b = bottom[1]->GetData();
			for (int idex = 0; idex < count; idex++) {
				top_data[idex] = bottom_data_a[idex] > bottom_data_b[idex] ?
					bottom_data_a[idex] : bottom_data_b[idex];
				mask[idex] = bottom_data_a[idex] > bottom_data_b[idex] ? 0 : 1;
			}
			// bottom 2 & 2++
			for (int blob_idex = 2; blob_idex < bottom.size(); blob_idex++) {
				bottom_data_b = bottom[blob_idex]->GetData();
				for (int idex = 0; idex < count; idex++) {
					if (bottom_data_b[idex] > top_data[idex]) {
						top_data[idex ] = bottom_data_b[idex];
						mask[idex] = blob_idex;
					}
				}
			}
			break;
		default:
			//std::cout << "Unknown elemenwise operation." << std::endl;
			break;
		}
	}
	
	template class JEltwiseLayer<float>;
	template class JEltwiseLayer<double>;

	REGISTER_LAYER_CLASS(Eltwise);

} // namespace jaffe
