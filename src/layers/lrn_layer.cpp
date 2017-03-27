#include "layers/lrn_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	void JLRNLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		LRNParameter lrn_param = this->m_layer_param.lrn_param();
		m_size = lrn_param.local_size();
		m_pre_pad = (m_size - 1) / 2;
		m_alpha = lrn_param.alpha();
		m_beta = lrn_param.beta();
		m_k = lrn_param.k();

		if (lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
			// 配置Split Layer的输入输出1
			m_split_top_vec.clear();
			m_split_top_vec.push_back(&m_product_input);
			m_split_top_vec.push_back(&m_square_input);
			LayerParameter split_param;
			m_split_layer.reset(new JSplitLayer<Dtype>(split_param));
			m_split_layer->SetUp(bottom, m_split_top_vec);
			// 配置Square Layer的输入输出
			m_square_bottom_vec.clear();
			m_square_top_vec.clear();
			m_square_bottom_vec.push_back(&m_square_input);
			m_square_top_vec.push_back(&m_square_output);
			LayerParameter square_param;
			square_param.mutable_power_param()->set_power(Dtype(2));
			m_square_layer.reset(new JPowerLayer<Dtype>(square_param));
			m_square_layer->SetUp(m_square_bottom_vec, m_square_top_vec);
			// 配置Pooling Layer的输入输出
			m_pool_top_vec.clear();
			m_pool_top_vec.push_back(&m_pool_output);
			LayerParameter pool_param;
			pool_param.mutable_pooling_param()->set_pool(
				PoolingParameter_PoolMethod_AVE);
			pool_param.mutable_pooling_param()->set_pad(m_pre_pad);
			pool_param.mutable_pooling_param()->set_kernel_size(m_size);
			m_pool_layer.reset(new JPoolingLayer<Dtype>(pool_param));
			m_pool_layer->SetUp(m_square_top_vec, m_pool_top_vec);
			// 配置Power Layer的输入输出
			m_power_top_vec.clear();
			m_power_top_vec.push_back(&m_power_output);
			LayerParameter power_param;
			power_param.mutable_power_param()->set_power(-m_beta);
			power_param.mutable_power_param()->set_scale(m_alpha);
			power_param.mutable_power_param()->set_shift(Dtype(1));
			m_power_layer.reset(new JPowerLayer<Dtype>(power_param));
			m_power_layer->SetUp(m_pool_top_vec, m_power_top_vec);
			// 配置Product Layer的输入输出
			m_product_bottom_vec.clear();
			m_product_bottom_vec.push_back(&m_product_input);
			m_product_bottom_vec.push_back(&m_power_output);
			LayerParameter product_param;
			product_param.mutable_eltwise_param()->set_operation(
				EltwiseParameter_EltwiseOp_PROD);
			m_product_layer.reset(new JEltwiseLayer<Dtype>(product_param));
			m_product_layer->SetUp(m_product_bottom_vec, top);
		}
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		m_num = bottom[0]->GetNum();
		m_channels = bottom[0]->GetChannels();
		m_height = bottom[0]->GetHeight();
		m_width = bottom[0]->GetWidth();
		switch (this->m_layer_param.lrn_param().norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			top[0]->Reshape(m_num, m_channels, m_height, m_width);
			m_scale.Reshape(m_num, m_channels, m_height, m_width);
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
			m_split_layer->Reshape(bottom, m_split_top_vec);
			m_square_layer->Reshape(m_square_bottom_vec, m_square_top_vec);
			m_pool_layer->Reshape(m_square_top_vec, m_pool_top_vec);
			m_power_layer->Reshape(m_pool_top_vec, m_power_top_vec);
			m_product_layer->Reshape(m_product_bottom_vec, top);
			break;
		}
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		switch (this->m_layer_param.lrn_param().norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			CrossChannelForward(bottom, top);
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
			WithinChannelForward(bottom, top);
			break;
		default:
			break;
		}
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::CrossChannelForward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();

		Dtype* scale_data = m_scale.GetMutableData();
		// 先常量
		for (int i = 0; i < m_scale.GetCount(); i++) {
			scale_data[i] = m_k;
		}
		JBlob<Dtype> padded_square(1, m_channels + m_size - 1, m_height, m_width);
		Dtype* padded_square_data = padded_square.GetMutableData();
		JaffeSet(padded_square.GetCount(), Dtype(0), padded_square_data);
		Dtype alpha_over_size = m_alpha / m_size;
		// 遍历图片
		for (int n = 0; n < m_num; n++) {
			// 计算 padded square
			JaffeSqr(m_channels * m_height * m_width, 
				bottom_data + bottom[0]->GetOffset(n),
				padded_square_data + padded_square.GetOffset(0, m_pre_pad));
			// 创建第一个缩放通道
			for (int c = 0; c < m_size; c++) 
				JaffeAxpy<Dtype>(m_height * m_width, alpha_over_size,
					padded_square_data + padded_square.GetOffset(0, c),
					scale_data + m_scale.GetOffset(n, 0));
			for ( int c = 1; c < m_channels; c++ ) {
				// 复制前一个缩放
				JaffeCopy<Dtype>(m_height * m_width, 
					scale_data + m_scale.GetOffset(n, c - 1),
					scale_data + m_scale.GetOffset(n, c));
				// add head
				JaffeAxpy<Dtype>(m_height * m_width, alpha_over_size,
					padded_square_data + padded_square.GetOffset(0, c + m_size - 1),
					scale_data + m_scale.GetOffset(n, c));
				// subtract tail
				JaffeAxpy<Dtype>(m_height * m_width, -alpha_over_size,
					padded_square_data + padded_square.GetOffset(0, c - 1),
					scale_data + m_scale.GetOffset(n, c));
			}
		}
		// 最后计算输出
		JaffePowx<Dtype>(m_scale.GetCount(), scale_data, -m_beta, top_data);
		JaffeMul<Dtype>(m_scale.GetCount(), top_data, bottom_data, top_data);
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::WithinChannelForward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		m_split_layer->LayerForward(bottom, m_split_top_vec);
		m_square_layer->LayerForward(m_square_bottom_vec, m_square_top_vec);
		m_pool_layer->LayerForward(m_square_top_vec, m_pool_top_vec);
		m_power_layer->LayerForward(m_pool_top_vec, m_power_top_vec);
		m_product_layer->LayerForward(m_product_bottom_vec, top);
	}

	template class JLRNLayer <double>;
	template class JLRNLayer <float>;

#ifdef CPU_ONLY
	STUB_GPU(JLRNLayer);
#endif
	REGISTER_LAYER_CLASS(LRN);

} // namespace jaffe
