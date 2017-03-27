#ifndef INPUT_LAYER_H_H
#define INPUT_LAYER_H_H

#include "layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JInputLayer : public JLayer<Dtype> {
	public:
		explicit JInputLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {}
		virtual inline const char* GetType() const {
			return "Input";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 0;
		}
		virtual inline int MinTopBlobs() const {
			return 1;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {}

		int m_batch;

	}; // class JInputLayer
} // namespace jafe
#endif
