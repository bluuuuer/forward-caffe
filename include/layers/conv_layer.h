#ifndef CONV_LAYER_H_H
#define CONV_LAYER_H_H

#include "layers/base_conv_layer.h"

namespace jaffe {

	template <typename Dtype>
	class JConvolutionLayer : public JBaseConvolutionLayer<Dtype> {
   	public:
		JConvolutionLayer(const LayerParameter& param)
			: JBaseConvolutionLayer<Dtype>(param) {}
		virtual inline const char* GetType() const { 
	   		return "Convolution";
		}
	
	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline bool ReverseDimensions() {
			return false;
		}
		virtual void ComputeOutputShape();
	};
} // namespace jaffe
#endif
