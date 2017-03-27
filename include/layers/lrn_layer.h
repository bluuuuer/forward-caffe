#ifndef LRN_LAYER_H_H
#define LRN_LAYER_H_H

#include "layer.h"
#include "layers/eltwise_layer.h"
#include "layers/power_layer.h"
#include "layers/pooling_layer.h"
#include "layers/split_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JLRNLayer : public JLayer<Dtype> {
	public:
		explicit JLRNLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "LRN";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 1;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		virtual void CrossChannelForward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top); // caffe CrossChannelForward_cpu()
		virtual void WithinChannelForward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
#ifndef CPU_ONLY
		virtual void CrossChannelForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
#endif

		int m_size;
		int m_pre_pad;
		Dtype m_alpha;	// 缩放因子，默认值为1
		Dtype m_beta; // 指数项，默认为5
		Dtype m_k;
		int m_num;
		int m_channels;
		int m_height;
		int m_width;

		JBlob<Dtype> m_scale;

		shared_ptr<JSplitLayer<Dtype>> m_split_layer;
		vector<JBlob<Dtype>*> m_split_top_vec;

		shared_ptr<JPowerLayer<Dtype>> m_square_layer;
		JBlob<Dtype> m_square_input;
		JBlob<Dtype> m_square_output;
		vector<JBlob<Dtype>*> m_square_bottom_vec;
		vector<JBlob<Dtype>*> m_square_top_vec;

		shared_ptr<JPoolingLayer<Dtype>> m_pool_layer;
		JBlob<Dtype> m_pool_output;
		vector<JBlob<Dtype>*> m_pool_top_vec;

		shared_ptr<JPowerLayer<Dtype>> m_power_layer;
		JBlob<Dtype> m_power_output;
		vector<JBlob<Dtype>*> m_power_top_vec;

		shared_ptr<JEltwiseLayer<Dtype>> m_product_layer;
		JBlob<Dtype> m_product_input;
		vector<JBlob<Dtype>*> m_product_bottom_vec;
	}; // class JLRNLayer
} // namespace jaffe
#endif
