// huangshize 2016.04.10
// 增加了构造函数

#ifndef NEURON_LAYER_H_H
#define NEURON_LAYER_H_H

#include "layer.h"

namespace jaffe {
	 
	template <typename Dtype>
	class JNeuronLayer : public JLayer<Dtype>{
	public:
		explicit JNeuronLayer(const LayerParameter& param)
	   		: JLayer<Dtype>(param) {}
		// hsz0407

		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int ExactNumTopBottom() const {
			return 1;
		}
	}; // class JNeruonLayer

} // namespace jaffe

#endif
