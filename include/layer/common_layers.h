#ifndef COMMON_LAYERS_H_H
#define COMMON_LAYERS_H_H

#include "layer.h"
#include "softmax_param.h"
#include "innerproduct_param.h"

namespace jaffe {

	template <typename Dtype>
	class JSoftmaxLayer : public JLayer<Dtype>{
	public:
		JSoftmaxLayer(){
			m_param = new JSoftmaxParam;
		};
		~JSoftmaxLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param); 
		bool SetParam(const vector<string> param);
		bool ReadParam();
		virtual bool Show();

	private:
		int m_outer_num;
		int m_inner_num;
		int softmax_axis;
//		Blob<Dtype> m_sum_multiplier;
//		Blob<Dtype> m_scale;

		JSoftmaxParam* m_param;
	}; // class JSoftmaxLayer

	template <typename Dtype>
	class JInnerProductLayer : public JLayer<Dtype>{
	public:
		JInnerProductLayer(){
			m_param = new JInnerProductParam;
		};
		~JInnerProductLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);
		bool SetParam(const vector<string> param);
		virtual bool Show();
		//virtual void Forward(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);

	private:
		int M_;
		int K_;
		int N_;
		bool bias_term;
		//Blob<Dtype> bias_multiplier;
		JInnerProductParam* m_param;
	}; // class JInnerProductLayer

} // namespace jaffe
#endif