#include "layers/pooling_layer.h"

namespace jaffe{
	
	template <typename Dtype>
	void JPoolingLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		PoolingParameter pool_param = this->m_layer_param.pooling_param();
		m_global_pooling = pool_param.global_pooling();
		if (m_global_pooling) {
			m_kernel_h = bottom[0]->GetHeight();
			m_kernel_w = bottom[0]->GetWidth();
		}
		else {
			if (pool_param.has_kernel_size())
				m_kernel_h = m_kernel_w = pool_param.kernel_size();
			else {
				m_kernel_h = pool_param.kernel_h();
				m_kernel_w = pool_param.kernel_w();
			}
		}
		if (!pool_param.has_pad_h())
			m_pad_h = m_pad_w = pool_param.pad();
		else {
			m_pad_h = pool_param.pad_h();
			m_pad_w = pool_param.pad_w();
		}
		if (!pool_param.has_stride_h())
			m_stride_h = m_stride_w = pool_param.stride();
		else {
			m_stride_h = pool_param.stride_h();
			m_stride_w = pool_param.stride_w();
		}
	}

	template <typename Dtype>
	void JPoolingLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		m_channels = bottom[0]->GetChannels();
		m_height = bottom[0]->GetHeight();
		m_width = bottom[0]->GetWidth();
		if (m_global_pooling) {
			m_kernel_h = bottom[0]->GetHeight();
			m_kernel_w = bottom[0]->GetWidth();
		}
		m_pooled_height = static_cast<int>(ceil(static_cast<float>(m_height + 2 * 
			m_pad_h - m_kernel_h) / m_stride_h)) + 1;
		m_pooled_width = static_cast<int>(ceil(static_cast<float>(m_width + 2 * 
			m_pad_w - m_kernel_w) / m_stride_w)) + 1;
		if (m_pad_h || m_pad_w) {
			if ((m_pooled_height - 1) * m_stride_h >= m_height + m_pad_h)
				m_pooled_height--;
			if ((m_pooled_width - 1) * m_stride_w >= m_width + m_pad_w) 
				m_pooled_width--;
		}	
		top[0]->Reshape(bottom[0]->GetNum(), m_channels, m_pooled_height, 
			m_pooled_width);
		if (top.size() > 1) 
			top[1]->ReshapeLike(*top[0]);
		// 如果是max pooling
		if (this->m_layer_param.pooling_param().pool() ==
			PoolingParameter_PoolMethod_MAX && top.size() == 1)
			m_max_idx.Reshape(bottom[0]->GetNum(), m_channels, m_pooled_height,
				m_pooled_width);
	}

	template <typename Dtype>
	void JPoolingLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
		const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetData();
		Dtype* top_data = top[0]->GetMutableData();
		const int top_count = top[0]->GetCount();
		// 当 top 的维度大于 1 时，会输出 mask 到 top[1]中
		const bool use_top_mask = top.size() > 1;
		int* mask = NULL;
		Dtype* top_mask = NULL;
		// 对于不同的 pooling 方法，在最外层使用 switch
		switch (this->m_layer_param.pooling_param().pool()) {
		case PoolingParameter_PoolMethod_MAX:
			// 初始化
			if (use_top_mask) {
				top_mask = top[1]->GetMutableData();
				JaffeSet(top_count, Dtype(-1), top_mask);
			}
			else {
				mask = m_max_idx.GetMutableData();
				JaffeSet(top_count, -1, mask);
			}
			JaffeSet(top_count, Dtype(-FLT_MAX), top_data);
			
			// 主循环
			for (int n = 0; n < bottom[0]->GetNum(); n++)
				for (int c = 0; c < m_channels; c++) {
					for (int ph = 0; ph < m_pooled_height; ph++)
						for (int pw = 0; pw < m_pooled_width; pw++) {
							int hstart = ph * m_stride_h - m_pad_h;
							int wstart = pw * m_stride_w - m_pad_w;
							int hend = min(hstart + m_kernel_h, m_height);
							int wend = min(wstart + m_kernel_w, m_width);
							hstart = max(hstart, 0);
							wstart = max(wstart, 0);
							const int pool_index = ph * m_pooled_width + pw;
							for (int h = hstart; h < hend; h++)
								for (int w = wstart; w < wend; w++) {
									const int index = h * m_width + w;
									if (bottom_data[index] > top_data[pool_index]) {
										top_data[pool_index] = bottom_data[index];
										if (use_top_mask) 
											top_mask[pool_index] = 
												static_cast<Dtype>(index);
										else
											mask[pool_index] = index;
									}
								}		
						}
					bottom_data += bottom[0]->GetOffset(0, 1);
					top_data += top[0]->GetOffset(0, 1);
					if (use_top_mask)
						top_mask += top[0]->GetOffset(0, 1);
					else
						mask += top[0]->GetOffset(0, 1);
				}	
			break;
		case PoolingParameter_PoolMethod_AVE:
			for (int i = 0; i < top_count; i++)
				top_data[i] = 0;
			// 主循环
			for (int n = 0; n < bottom[0]->GetNum(); n++)
				for (int c = 0; c < m_channels; c++) {
					for (int ph = 0; ph < m_pooled_height; ph++)
						for (int pw = 0; pw < m_pooled_width; pw++) {
							int hstart = ph * m_stride_h - m_pad_h;
							int wstart = pw * m_stride_w - m_pad_w;
							int hend = min(hstart + m_kernel_h, m_height + m_pad_h);
							int wend = min(wstart + m_kernel_w, m_width + m_pad_w);
							int pool_size = (hend - hstart) * (wend - wstart);
							hstart = max(hstart, 0);
							wstart = max(wstart, 0);
							hend = min(hend, m_height);
							wend = min(wend, m_width);
							for (int h = hstart; h < hend; h++)
								for (int w = wstart; w < wend; w++)
									top_data[ph * m_pooled_width + pw] += 
										bottom_data[h * m_width + w];
							top_data[ph * m_pooled_width + pw] /= pool_size;
						}	
				bottom_data += bottom[0]->GetOffset(0, 1);
				top_data += top[0]->GetOffset(0, 1);
				}	
			break;
		case PoolingParameter_PoolMethod_STOCHASTIC:
			// 待补充
			break;
		default:
			cout << "ERROR: Unknown pooling method." << endl;
			break;
		}
	}

	template class JPoolingLayer <double>;
	template class JPoolingLayer <float>;
#ifdef CPU_ONLY
	STUB_GPU(JPoolingLayer);
#endif

	REGISTER_LAYER_CLASS(Pooling);

} // namespace jaffe
