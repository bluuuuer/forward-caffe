#ifndef NEURON_LAYERS_H_H
#define NEURON_LAYERS_H_H

#include "layer.h"
#include "relu_param.h"
#include "dropout_param.h"

namespace jaffe {
	 
	template <typename Dtype>
	class JNeuronLayer : public JLayer<Dtype>{
	public:
		JNeuronLayer(){};
		~JNeuronLayer(){};
		//virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);
	}; // class JNeruonLayer

	template <typename Dtype>
	class JReLULayer : public JNeuronLayer<Dtype>{
	public:
		JReLULayer(){
			m_param = new JReLUParam;
		};
		~JReLULayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);
		bool SetParam(const vector<string> param);
		bool Show();

	private:
		JReLUParam* m_param;
	}; // class JReLULayer

	template <typename Dtype>
	class JDropoutLayer : public JNeuronLayer<Dtype>{
	public:
		JDropoutLayer(){
			m_param = new JDropoutParam;
		};
		~JDropoutLayer(){
			delete[] m_param;
		};

		bool Init(const vector<string> param);
		bool SetParam(const vector<string> param);
		bool Show();

	private:
		//Blob<unsigned int> m_rand_vec;
		//Dtype m_threshold;
		//Dtype m_scale;
		unsigned int m_uint_thres;

		JDropoutParam* m_param;
	}; //

} // namespace jaffe

#endif