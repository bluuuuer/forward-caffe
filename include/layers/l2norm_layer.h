#ifndef L2NORM_LAYER_H_H
#define L2NORM_LAYER_H_H

#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	class JL2NormLayer : public JLayer<Dtype> {
	public:
		explicit JL2NormLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "L2Norm";
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


		JBlob<Dtype> m_norm, m_temp;
		JBlob<Dtype> m_sum_multiplier;
		Dtype m_eps;

	}; // class JL2NormLayer
} // namespace jaffe
#endif
