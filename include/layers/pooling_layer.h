#ifndef POOLING_LAYER_H_H
#define POOLING_LAYER_H_H

#include "layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JPoolingLayer : public JLayer<Dtype> {
	public:
		explicit JPoolingLayer(const LayerParameter& param) 
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		virtual inline const char* GetType() const {
			return "Pooling";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int MinTopBlobs() const {
			return 1;
		}
		// MAX POOL 层会输出一个额外的 mask top blob
		virtual inline int MaxTopBlobs() const {
			return (this->m_layer_param.pooling_param().pool() == 
				PoolingParameter_PoolMethod_MAX) ? 2 : 1;
		}
	
	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		int m_kernel_h, m_kernel_w;
		int m_stride_h, m_stride_w;
		int m_pad_h, m_pad_w;
		int m_channels;
		int m_height, m_width;
		int m_pooled_height, m_pooled_width;
		bool m_global_pooling;
		//JBlob<Dtype> m_rand_idx; // STOCHASTIC 还未实现
		JBlob<int> m_max_idx;

	}; // class JPoolingLayer
} // namespace jaffe
#endif
