#ifndef SILENCE_LAYER_H_H
#define SILENCE_LAYER_H_H

#include "layer.h"

namespace jaffe {
		
	template <typename Dtype>
	class JSilenceLayer : public JLayer<Dtype> {
	public:
		explicit JSilenceLayer(const LayerParameter param) : JLayer<Dtype>(param) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {}
		virtual inline const char* GetType() const {
			return "Silence";
		}
		virtual inline int MinBottomBlobs() const {
			return 1;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 0;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {}
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		
	}; // class JSilenceLayer

} // namespace jaffe
#endif
