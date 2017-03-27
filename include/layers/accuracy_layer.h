#ifndef ACCURACY_LAYER_H_H
#define ACCURACY_LAYER_H_H

#include "layers/loss_layer.h"

namespace jaffe {

	template <typename Dtype>
	class JAccuracyLayer : public JLayer<Dtype> {
	public:
		explicit JAccuracyLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline const char* GetType() const {
			return "Accuracy";
		}
		virtual inline int MinTopBlobs() const {
			return 1;
		}
		virtual inline int MaxTopBlobs() const {
			return 2;
		}

	protected:
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

		int m_label_axis, m_outer_num, m_inner_num;
		int m_top_k;
		bool m_has_ignore_label;
		int m_ignore_label;
		JBlob<Dtype> m_nums_buffer;
	}; // class JAccuracyLayer
} // namespace jaffe
#endif
