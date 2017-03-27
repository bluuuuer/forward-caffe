#ifndef DECONV_LAYER_H_H
#define DECONV_LAYER_H_H

#include "layers/base_conv_layer.h"

namespace jaffe {

	template <typename Dtype>
	class JDeconvolutionLayer : public JBaseConvolutionLayer<Dtype> {
   	public:
		JDeconvolutionLayer(const LayerParameter& param)
			: JBaseConvolutionLayer<Dtype>(param) {}
		virtual inline const char* GetType() const { 
	   		return "Deconvolution";
		}
	
	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline bool ReverseDimensions() {
			return false;
		}
		virtual void ComputeOutputShape();
	};
} // namespace jaffe
#endif
