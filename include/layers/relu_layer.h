#ifndef RELU_LAYER_H_H
#define RELU_LAYER_H_H

#include "layers/neuron_layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JReLULayer : public JNeuronLayer<Dtype> {
	public:
		explicit JReLULayer(const LayerParameter& param) 
				: JNeuronLayer<Dtype>(param) {}
		virtual inline const char* GetType() const {
			return "ReLU";
		}	
	
	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
	}; // class JReLULayer
} // namespace jaffe
#endif
