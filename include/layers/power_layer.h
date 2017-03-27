#ifndef POWER_LAYER_H_H
#define POWER_LAYER_H_H

#include "layers/neuron_layer.h"

namespace jaffe {

	template <typename Dtype>
	class JPowerLayer : public JNeuronLayer<Dtype> {
	public:
		explicit JPowerLayer(const LayerParameter& param) 
			: JNeuronLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "Power";
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom, 
			const vector<JBlob<Dtype>*>& top);

		Dtype m_power;
		Dtype m_scale;
		Dtype m_shift;
		Dtype m_diff_scale;
	}; // class JPowerLayer
} // namespace jaffe
#endif
