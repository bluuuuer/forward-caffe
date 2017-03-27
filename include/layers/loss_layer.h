// huangshize 2016.04.10
// 增加了构造函数

#ifndef LOSS_LAYER_H_H
#define LOSS_LAYER_H_H

#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	class JLossLayer : public JLayer<Dtype>{
	public:
		explicit JLossLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const {
			return 2;
		}
		virtual inline bool AutoTopBlobs() const {
			return true;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 1;
		}
	}; // class JLossLayer

} // namespace jaffe
#endif
