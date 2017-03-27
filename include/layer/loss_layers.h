#ifndef LOSS_LAYERS_H_H
#define LOSS_LAYERS_H_H

#include "layer.h"
#include "accuracy_param.h"

namespace jaffe {

	template <typename Dtype>
	class JAccuracyLayer : public JLayer<Dtype>{
	public:
		JAccuracyLayer(){
			m_param = new JAccuracyParam;
		};
		~JAccuracyLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);
		bool SetParam(const vector<string> param);
		virtual bool Show();

	private:
		int m_label_axis, m_outer_num, m_inner_num;
		int m_top_k;
		bool m_has_ignore_label;
		int m_ignore_label;
		//Blob<Dtype> m_nums_buffer;

		JAccuracyParam* m_param;
	}; // class JAccuracyLayer

} // namespace jaffe
#endif