#include "layers/inner_product_layer.h"
#include <sys/time.h>

namespace jaffe {

	template <typename Dtype>
	void JInnerProductLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		InnerProductParameter ip_param = this->m_layer_param.inner_product_param();
		const int num_output = ip_param.num_output();
		m_bias_term = ip_param.bias_term();
		m_transpose = ip_param.transpose();
		m_N = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(ip_param.axis());
		m_K = bottom[0]->GetCount(axis);
		if (this->m_blobs.size() > 0) {
			cout << __FILE__ << "\t" << __LINE__ 
				<< "\tSkipping parameter initialization" << endl;
		}
		else {
			if (m_bias_term)
				this->m_blobs.resize(2);
			else
				this->m_blobs.resize(1);
			// 初始化weights
			vector<int> weight_shape(2);
			if (m_transpose) {
				weight_shape[0] = m_K;
				weight_shape[1] = m_N;
			}
			else {
				weight_shape[0] = m_N;
				weight_shape[1] = m_K;
			}
			this->m_blobs[0].reset(new JBlob<Dtype>(weight_shape));
			// 填充weights
			shared_ptr<JFiller<Dtype> > weight_filler(GetFiller<Dtype>());
			weight_filler->Fill(this->m_blobs[0].get());
			if (m_bias_term) {
				vector<int> bias_shape(1, m_N);
				this->m_blobs[1].reset(new JBlob<Dtype>(bias_shape));
				shared_ptr<JFiller<Dtype> > bias_filler(GetFiller<Dtype>());
				bias_filler->Fill(this->m_blobs[1].get());
			}
		}
		this->m_param_propagate_down.resize(this->m_blobs.size(), true);
	}
	
	template <typename Dtype>
	void JInnerProductLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->m_layer_param.inner_product_param().axis());
		const int new_K = bottom[0]->GetCount(axis);
		m_M = bottom[0]->GetCount(0, axis);
		vector<int> top_shape = bottom[0]->GetShape();
		top_shape.resize(axis + 1);
		top_shape[axis] = m_N;
		top[0]->Reshape(top_shape);
		if (m_bias_term) {
			vector<int> bias_shape(1, m_M);
			m_bias_multiplier.Reshape(bias_shape);
			JaffeSet(m_M, Dtype(1), m_bias_multiplier.GetMutableData());
		}
	}

	template <typename Dtype>
	void JInnerProductLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();
		const Dtype* weight = this->m_blobs[0]->GetData();

		JaffeGemm<Dtype>(0, m_transpose ? 0 : 1, m_M, m_N, m_K, (Dtype)1.,
			bottom_data, weight, (Dtype)0., top_data);
		if (m_bias_term) {
			JaffeGemm<Dtype>(0, 0, m_M, m_N, 1, (Dtype)1., 
				m_bias_multiplier.GetData(), this->m_blobs[1]->GetData(), (Dtype)1., 
				top_data);
		}
	}

	template class JInnerProductLayer <double>;
	template class JInnerProductLayer <float>;
#ifdef CPU_ONLY
	STUB_GPU(JInnerProductLayer);
#endif

	REGISTER_LAYER_CLASS(InnerProduct);

} // namespace jaffe
