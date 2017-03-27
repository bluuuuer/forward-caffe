#ifndef VISION_LAYERS_H_H
#define VISION_LAYERS_H_H

#include "layer.h"
#include "convolution_param.h"
#include "pooling_param.h"
#include "lrn_param.h"

namespace jaffe {

	template <typename Dtype>
	class JBaseConvolutionLayer : public JLayer<Dtype>{
	public:
		JBaseConvolutionLayer(){};

		~JBaseConvolutionLayer(){};

	protected:
		//Blob<int> m_kernel_shape;
		//Blob<int> m_stride;
		//Blob<int> m_pad;
		//Blob<int> m_conv_input_shape;
		vector<int> m_col_buffer_shape;
		vector<int> m_output_shape;
		const vector<int>* m_bottom_shape;

		int m_num_spatial_axis;
		int m_bottom_dim;
		int m_top_dim;
		int m_channel_axis;
		int m_num;
		int m_channels;
		int m_group;
		int m_out_spatial_dim;
		int m_weight_offset;
		int m_num_output;
		bool m_bias_term;
		bool m_is_1x1;
		bool m_force_nd_im2col;

	private:
		int m_num_kernel_im2col;
		int m_num_kernel_col2im;
		int m_conv_out_channels;
		int m_conv_in_channels;
		int m_conv_out_spatial_dim;
		int m_kernel_dim;
		int m_col_offset;
		int m_output_offset;

//		Blob<Dtype> m_col_buffer;
//		Blob<Dtype> m_bias_multiplier;
	};

	template <typename Dtype>
	class JConvolutionLayer : public JBaseConvolutionLayer<Dtype>{
	public:
		JConvolutionLayer(){
			m_param = new JConvolutionParam;
		};

		~JConvolutionLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);

		bool SetParam(const vector<string> param);

		bool ReadParam();

		virtual bool Show();

	private:

		JConvolutionParam* m_param;

	};

	template <typename Dtype>
	class JPoolingLayer : public JLayer<Dtype> {
	public:
		JPoolingLayer(){
			m_param = new JPoolingParam;
		};

		~JPoolingLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);

		bool SetParam(const vector<string> param);

		bool ReadParam();

		virtual bool Show();

	private:
		JPoolingParam* m_param;

		int m_kernel_h, m_kernel_w;
		int m_stride_h, m_stride_w;
		int m_pad_h, m_pad_w;
		int m_channels;
		int m_height, m_width;
		int m_pooled_height, m_pooled_width;
		bool m_global_pooling;

//		Blob<Dtype> m_rand_idex;
//		Blob<int> max_idx;

	}; // class JPoolingLayer

	template <typename Dtype>
	class JLRNLayer : public JLayer<Dtype>{
	public:
		JLRNLayer(){
			m_param = new JLRNParam;
		};
		~JLRNLayer(){
			delete[] m_param;
		};
		
		bool Init(const vector<string> param);
		bool SetParam(const vector<string> param);
		bool ReadParam();
		virtual bool Show();

	private:
		JLRNParam* m_param;

		int m_size;
		int pre_pad;
		Dtype m_alpha;
		Dtype m_beta;
		Dtype m_k;
		int m_num;
		int m_channels;
		int m_height;
		int m_width;

		//Blob<Dtype> m_scale;

		//shared_ptr<SplitLayer<Dtype>> m_split_layer;
		//vector<Blob<Dtype>*> m_split_top_vec;
		//shared_ptr<PowerLayer<Dtype>> m_square_layer;
		//Blob<Dtype> m_square_input;
		//Blob<Dtype> m_square_output;
		//vector<Blob<Dtype>*> m_square_bottom_vec;
		//vector<Blob<Dtype>*> m_square_top_vec;
		//shared_ptr<PoolingLayer<Dtype>> m_pool_layer;
		//Blob<Dtype> m_power_output;
		//vecotr<Blob<Dtype>*> pool_top_vec;
		//shared_ptr<PowerLayer<Dtype>> m_power_layer;
		//Blob<Dtype> m_power_output;
		//vecotr<Blob<Dtype>*> m_power_top_vec;
		//shared_ptr<EltwiseLayer<Dtype>> m_product_layer;
		//Blob<Dtype> m_product_input;
		//vector<Blob<Dtype>*> m_product_bottom_vec;

	}; // JLRNLayer
}
#endif