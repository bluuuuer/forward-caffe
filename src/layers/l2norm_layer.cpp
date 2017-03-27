#include "layers/l2norm_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JL2NormLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {
		top[0]->Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), 
			bottom[0]->GetHeight(), bottom[0]->GetWidth());
		m_norm.Reshape(bottom[0]->GetNum(), 1, 1, 1);
		m_temp.Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), 
			bottom[0]->GetHeight(), bottom[0]->GetWidth());
		// Forward 用不到
		//m_temp1.ReshapeLike(*bottom[0]);
		m_sum_multiplier.Reshape(1, bottom[0]->GetChannels(), bottom[0]->GetHeight(),
			bottom[0]->GetWidth());
		Dtype* multiplier_data = m_sum_multiplier.GetMutableData();
		JaffeSet(m_sum_multiplier.GetCount(), Dtype(1), multiplier_data);
		m_eps = this->m_layer_param.l2norm_param().eps();
	}

	template <typename Dtype>
	void JL2NormLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();
		const int num = bottom[0]->GetNum();
		const int all_fea_count = bottom[0]->GetCount();
		const int dim = all_fea_count / num;
		// 先计算每个数的开方
		JaffeMul(all_fea_count, bottom_data, bottom_data, m_temp.GetMutableData());
		// 求同一个 sample 的 features 的平方和
		JaffeGemv<Dtype>(0, num, dim, 1, m_temp.GetData(), m_sum_multiplier.GetData(),
			0., m_norm.GetMutableData());
		// 开方
		JaffePowx(num, m_norm.GetData(), Dtype(0.5), m_norm.GetMutableData());
		// 加上 eps
		JaffeAddScalar(num, m_eps, m_norm.GetMutableData());
		// 得到最终的 norm 矩阵
		// 前面计算出来的 norm 矩阵是 n*1，为了计算方便， 我们需要得到 n*d 的 norm
		// 矩阵，其中同一行的 d 个元素值都一样，为该 sample 的归一化因子
		// 所以直接利用让 norm 矩阵乘以一个 1*d 的矩阵（全1）即可得到最终的 norm 矩阵
		JaffeGemm<Dtype>(0, 0, num, dim, 1, 1, m_norm.GetData(), 
			m_sum_multiplier.GetData(), 0., m_temp.GetMutableData());
		// 进行 norm
		JaffeDiv(all_fea_count, bottom_data, m_temp.GetData(), top_data);
	}

	template class JL2NormLayer<float>;
	template class JL2NormLayer<double>;
#ifdef CPU_ONLY
	STUB_GPU(JL2NormLayer);
#endif

	REGISTER_LAYER_CLASS(L2Norm);
} // namespace jaffe
