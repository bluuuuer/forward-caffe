#ifndef CONCAT_LAYER_H_H
#define CONCAT_LAYER_H_H


#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	class JConcatLayer : public JLayer<Dtype> {
 	public:
  		explicit JConcatLayer(const LayerParameter& param)
      		: JLayer<Dtype>(param) {}
  		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
      		const vector<JBlob<Dtype>*>& top);
  		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
      		const vector<JBlob<Dtype>*>& top);

  		virtual inline const char* GetType() const { 
			return "Concat"; 
		}
  		virtual inline int MinBottomBlobs() const { 
			return 1; 
		}
  		virtual inline int ExactNumTopBlobs() const { 
			return 1; 
		}

 	protected:
  		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
      		const vector<JBlob<Dtype>*>& top);
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);

  		int m_count;
  		int m_num_concats;
  		int m_concat_input_size;
  		int m_concat_axis;

	}; // class JConcatLayer

}  // namespace jaffe

#endif  // CONCAT_LAYER_H_H
