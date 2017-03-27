#ifndef FLATTEN_LAYER_H_H
#define FLATTEN_LAYER_H_H

#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	class JFlattenLayer : public JLayer<Dtype> {
	public:
		explicit JFlattenLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
				const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "Flatten";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int ExactNumTopBlobs() const {
			return 1;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

	}; // class JFlattenLayer
} // namespace jaffe
#endif
