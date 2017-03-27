#ifndef SPLIT_LAYER_H_H
#define SPLIT_LAYER_H_H

#include "layer.h"

namespace jaffe {
	
	template <typename Dtype>
	class JSplitLayer : public JLayer<Dtype> {
	public:
		explicit JSplitLayer(const LayerParameter& param) 
			: JLayer<Dtype>(param) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "Split";
		}
		virtual inline int ExactNumBottomBlobs() const {
			return 1;
		}
		virtual inline int MinTopBlobs() const {
			return 1;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		//int m_count; // Forward时用不到
	}; // class JSplitLayer
} // namespace jaffe
#endif
