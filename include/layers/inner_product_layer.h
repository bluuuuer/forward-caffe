#ifndef INNER_PRODUCT_LAYER_H_H
#define INNER_PRODUCT_LAYER_H_H

#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	class JInnerProductLayer : public JLayer<Dtype> {
	public:
		explicit JInnerProductLayer(const LayerParameter& param) 
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const { 
			return "InnerProduct";
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
		
		int m_M;
		int m_K;
		int m_N;
		bool m_bias_term;
		JBlob<Dtype> m_bias_multiplier;
		bool m_transpose;
	}; // class JInneerProductLayer
} // namespace jaffe
#endif
