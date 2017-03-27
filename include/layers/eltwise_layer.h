#ifndef ELTWISE_LAYER_H_H
#define ELTWISE_LAYER_H_H

#include "layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JEltwiseLayer : public JLayer<Dtype> {
	public:
		explicit JEltwiseLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* type() const { 
			return "Eltwise";   	
		}
		virtual inline int MinBottomBlobs() const {
			return 2;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 1;
		}
		
	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		
		EltwiseParameter_EltwiseOp m_op;
		vector<Dtype> m_coeffs;
		JBlob<int> m_max_idx;

		//bool m_stable_prod_grad; // Forward 用不到
	}; // class JEltwiseLayer
} // namespace jaffe
#endif
