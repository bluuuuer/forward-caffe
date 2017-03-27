#ifndef DROPOUT_LAYER_H_H
#define DROPOUT_LAYER_H_H

#include "layers/neuron_layer.h"

namespace jaffe {

	template <typename Dtype>
	class JDropoutLayer : public JNeuronLayer<Dtype> {
	public:
		explicit JDropoutLayer(const LayerParameter& param) 
			: JNeuronLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* type() const { 
	   		return "Dropout";
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		//------------- test forward 阶段不需要这些参数
		//JBlob<unsigned<int> m_rand_vec;
		//Dtype m_threshold; // drop 某个输入的概率
		//Dtype m_scale;
		//unsigned int m_uint_thres;
	}; // class JDropoutLayer
} // namespace jaffe
#endif
