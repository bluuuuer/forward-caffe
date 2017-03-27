#ifndef SOFTMAX_LAYER_H_H
#define SOFTMAX_LAYER_H_H

#include "layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JSoftmaxLayer : public JLayer<Dtype> {
	public:
		explicit JSoftmaxLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "Softmax";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 1;
		}

	private:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		int m_outer_num;
		int m_inner_num;
		int m_softmax_axis;
		JBlob<Dtype> m_sum_multiplier;
		JBlob<Dtype> m_scale; // 暂存计算结果
	};	// class JSoftmaxLayer
} // namespace caffe
#endif
