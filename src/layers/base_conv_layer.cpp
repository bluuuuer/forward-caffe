////
// Created by bluuuuer on 16-4-10.
//

#include "layers/base_conv_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		ConvolutionParameter conv_param = this->m_layer_param.convolution_param();
		m_force_nd_im2col = conv_param.force_nd_im2col();
		m_channel_axis = bottom[0]->CanonicalAxisIndex(conv_param.axis());
		const int first_spatial_axis = m_channel_axis + 1;
		const int num_axes = bottom[0]->GetNumAxes();
		m_num_spatial_axes = m_channel_axis + 1;
		vector<int> bottom_dim_blob_shape(1, m_num_spatial_axes + 1);
		vector<int> spatial_dim_blob_shape(1, max(m_num_spatial_axes, 1));
		// filter kernel dimensions
		m_kernel_shape.Reshape(spatial_dim_blob_shape);
		int* kernel_shape_data = m_kernel_shape.GetMutableData();
		if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
			kernel_shape_data[0] = conv_param.kernel_h();
			kernel_shape_data[1] = conv_param.kernel_w();
		}
		else
			for (int i = 0; i < m_num_spatial_axes; i++)
				kernel_shape_data[i] = 
					conv_param.kernel_size((conv_param.kernel_size_size() == 1) 
						? 0 : i);
		// stride dimensions
		m_stride.Reshape(spatial_dim_blob_shape);
		int* stride_data = m_stride.GetMutableData();
		if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
			stride_data[0] = conv_param.stride_h();
			stride_data[1] = conv_param.stride_w();
		}
		else
			for (int i = 0; i < m_num_spatial_axes; i++)
				stride_data[i] = (conv_param.stride_size() == 0) ? 1 : 
					conv_param.stride((conv_param.stride_size() == 1) ? 0 : i);
		// pad dimensions
		m_pad.Reshape(spatial_dim_blob_shape);
		int* pad_data = m_pad.GetMutableData();
		if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
			pad_data[0] = conv_param.pad_h();
			pad_data[1] = conv_param.pad_w();
		}
		else
			for (int i = 0; i < m_num_spatial_axes; i++) 
				pad_data[i] = (conv_param.pad_size() == 0) ? 0 :
					conv_param.pad((conv_param.pad_size() == 1) ? 0 : i);
		// dilation dimensions
		m_dilation.Reshape(spatial_dim_blob_shape);
		int* dilation_data = m_dilation.GetMutableData();
		for (int i = 0; i < m_num_spatial_axes; i++)
			dilation_data[i] = (conv_param.dilation_size() == 0) ? 1 :
				conv_param.dilation((conv_param.dilation_size() == 1) ? 0 : 1);
		// special case: im2col is the identity for 1x1 convolution with stride 1
		// and no padding , so flag for skipping the buffer and transformation
		m_is_1x1 = true;
		for (int i = 0; i < m_num_spatial_axes; i++) {
			m_is_1x1 &= 
				kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
			if (!m_is_1x1)
				break;
		}
		//-------配置输出的通道和group
		m_channels = bottom[0]->GetShape(m_channel_axis);
		m_num_output = conv_param.num_output();
		m_group = conv_param.group();
		if (ReverseDimensions()) {
			m_conv_out_channels = m_channels;
			m_conv_in_channels = m_num_output;
		}
		else {
			m_conv_out_channels = m_num_output;
			m_conv_in_channels = m_channels;
		}
		//------配置权重weights和偏置biases
		vector<int> weight_shape(2);
		weight_shape[0] = m_conv_out_channels;
		weight_shape[1] = m_conv_in_channels / m_group;
		for (int i = 0; i < m_num_spatial_axes; i++) 
			weight_shape.push_back(kernel_shape_data[i]);
		m_bias_term = conv_param.bias_term();
		vector<int> bias_shape(m_bias_term, m_num_output);
		if (this->m_blobs.size() > 0) {
			if (weight_shape != this->m_blobs[0]->GetShape())
				JBlob<Dtype> weight_shaped_blob(weight_shape);
			if (m_bias_term && bias_shape != this->m_blobs[1]->GetShape())
				JBlob<Dtype> bias_shaped_blob(bias_shape);		
		}
		else { 
			if (m_bias_term) 
				this->m_blobs.resize(2);
			else
				this->m_blobs.resize(1);
//			 //初始化权重的值
			this->m_blobs[0].reset(new JBlob<Dtype>(weight_shape));
			//=======改为constant填充=======
			shared_ptr<JFiller<Dtype> > weight_filler(GetFiller<Dtype>());
			weight_filler->Fill(this->m_blobs[0].get());
			if (m_bias_term) {
				this->m_blobs[1].reset(new JBlob<Dtype>(bias_shape));
				//======改为constant填充======
				shared_ptr<JFiller<Dtype> > bias_filler(GetFiller<Dtype>());
				bias_filler->Fill(this->m_blobs[1].get());
			}
		}
		m_kernel_dim = this->m_blobs[0]->GetCount(1);
		m_weight_offset = m_conv_out_channels * m_kernel_dim / m_group;
		this->m_param_propagate_down.resize(this->m_blobs.size(), true);
	}

	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const int first_spatial_axis = m_channel_axis + 1;
		m_num = bottom[0]->GetCount(0, m_channel_axis);
		m_bottom_shape = &bottom[0]->GetShape();
		ComputeOutputShape();
		vector<int> top_shape(bottom[0]->GetShape().begin(), 
			bottom[0]->GetShape().begin() + m_channel_axis);
		top_shape.push_back(m_num_output);
		for (int i = 0; i < m_num_spatial_axes; i++)
			top_shape.push_back(m_output_shape[i]);
		for (int i = 0; i < top.size(); i++)
			top[i]->Reshape(top_shape);
		if (ReverseDimensions())
			m_conv_out_spatial_dim = bottom[0]->GetCount(first_spatial_axis);
		else
			m_conv_out_spatial_dim = top[0]->GetCount(first_spatial_axis);
		m_col_offset = m_kernel_dim * m_conv_out_spatial_dim;
		m_output_offset = m_conv_out_channels * m_conv_out_spatial_dim / m_group;
		// input dimensions
		vector<int> bottom_dim_blob_shape(1, m_num_spatial_axes + 1);
		m_conv_input_shape.Reshape(bottom_dim_blob_shape);
		int* conv_input_shape_data = m_conv_input_shape.GetMutableData();
		for (int i = 0; i < m_num_spatial_axes + 1; i++) {
			if (ReverseDimensions())
				conv_input_shape_data[i] = top[0]->GetShape(m_channel_axis + i);
			else
				conv_input_shape_data[i] = bottom[0]->GetShape(m_channel_axis + i);
		}
		for (int i = 0; i < m_num_spatial_axes + 1; i++) {
		}
		//
		m_col_buffer_shape.clear();
		m_col_buffer_shape.push_back(m_kernel_dim * m_group);
		for (int i = 0; i < m_num_spatial_axes; i++) {
			if (ReverseDimensions())
				m_col_buffer_shape.push_back(GetInputShape(i + 1));
			else 
				m_col_buffer_shape.push_back(m_output_shape[i]);
		}
		m_col_buffer.Reshape(m_col_buffer_shape);
		m_bottom_dim = bottom[0]->GetCount(m_channel_axis);
		m_top_dim = top[0]->GetCount(m_channel_axis);
		m_num_kernels_im2col = m_conv_in_channels * m_conv_out_spatial_dim;
		m_num_kernels_col2im = ReverseDimensions() ? m_top_dim : m_bottom_dim;
		// 初始化bias_multiplier全为1
		m_out_spatial_dim = top[0]->GetCount(first_spatial_axis);
		if (m_bias_term) {
			vector<int> bias_multiplier_shape(1, m_out_spatial_dim);
			m_bias_multiplier.Reshape(bias_multiplier_shape);
			JaffeSet(m_bias_multiplier.GetCount(), Dtype(1), 
				m_bias_multiplier.GetMutableData());
		}
	}

    template <typename Dtype>
    void JBaseConvolutionLayer<Dtype>::ConvIm2col(const Dtype* data,
                                                Dtype* col_buff){
        if (!m_force_nd_im2col && m_num_spatial_axes == 2){
            Im2col(data, m_conv_in_channels, m_conv_input_shape.GetData()[1],
                m_conv_input_shape.GetData()[2], m_kernel_shape.GetData()[0],
                m_kernel_shape.GetData()[1], m_pad.GetData()[0], m_pad.GetData()[1],
                m_stride.GetData()[0], m_stride.GetData()[1], m_dilation.GetData()[0],
				m_dilation.GetData()[1], col_buff);
        }
        else{
            Im2colNd(data, m_num_spatial_axes, m_conv_input_shape.GetData(),
                m_col_buffer_shape.data(), m_kernel_shape.GetData(),
                m_pad.GetData(), m_stride.GetData(), m_dilation.GetData(), col_buff);
        }
    }

	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::ConvCol2im(const Dtype* col_buff,
			Dtype* data) {
		if (!m_force_nd_im2col && m_num_spatial_axes == 2) {
			Col2im(col_buff, m_conv_in_channels, m_conv_input_shape.GetData()[1],
				m_conv_input_shape.GetData()[2], m_kernel_shape.GetData()[0], 
				m_kernel_shape.GetData()[1], m_pad.GetData()[0], m_pad.GetData()[1],
				m_stride.GetData()[0], m_stride.GetData()[1], m_dilation.GetData()[0],
				m_dilation.GetData()[1], data);
		} else {
			Col2imNd(col_buff, m_num_spatial_axes, m_conv_input_shape.GetData(),
				m_col_buffer_shape.data(), m_kernel_shape.GetData(), m_pad.GetData(),
				m_stride.GetData(), m_dilation.GetData(), data);
		}
	}

#ifndef CPU_ONLY
	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::ConvIm2colGpu(const Dtype* data, 
					Dtype* col_buff) {
		if (!m_force_nd_im2col && m_num_spatial_axes == 2) {
			Im2colGpu(data, m_conv_in_channels, m_conv_input_shape.GetData()[1],
				m_conv_input_shape.GetData()[2], m_kernel_shape.GetData()[0],
				m_kernel_shape.GetData()[1], m_pad.GetData()[0], m_pad.GetData()[1],
				m_stride.GetData()[0], m_stride.GetData()[1],
				m_dilation.GetData()[0], m_dilation.GetData()[1], col_buff);	
		} else {
			Im2colNdGpu(data, m_num_spatial_axes, m_num_kernels_im2col,
				m_conv_input_shape.GetGpuData(), m_col_buffer.GetGpuShape(),
				m_kernel_shape.GetGpuData(), m_pad.GetGpuData(), 
				m_stride.GetGpuData(), m_dilation.GetGpuData(), col_buff);
		}
	}

#endif

    template <typename Dtype>
    void JBaseConvolutionLayer<Dtype>::ForwardGemm(const Dtype* input,
    	const Dtype* weights, Dtype* output, bool skip_im2col) {
        const Dtype* col_buff = input;
        if (!m_is_1x1){
            if (!skip_im2col){
                ConvIm2col(input, m_col_buffer.GetMutableData());
            }
            col_buff = m_col_buffer.GetData();
        }
        for(int g = 0; g < m_group; g++){
           	JaffeGemm<Dtype>(0, 0, m_conv_out_channels / m_group, 
				m_conv_out_spatial_dim, m_kernel_dim, (Dtype)1., 
				weights + m_weight_offset * g, col_buff + m_col_offset * g, 
				(Dtype)0., output + m_output_offset * g);
        }
    }

    template <typename Dtype>
    void JBaseConvolutionLayer<Dtype>::ForwardBias(Dtype* output,
                                                   const Dtype* bias) {
        JaffeGemm<Dtype>(0, 0, m_num_output, m_out_spatial_dim, 1, (Dtype)1., bias, 
			m_bias_multiplier.GetData(), (Dtype)1., output);
    }

	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::BackwardGemm(const Dtype* output, 
			const Dtype* weights, Dtype* input) {
		Dtype* col_buff = m_col_buffer.GetMutableData();
		if (m_is_1x1) {
			col_buff = input;
		}
		for (int g = 0; g < m_group; g++) {

			JaffeGemm<Dtype>(1, 0, m_kernel_dim, m_conv_out_spatial_dim, 
				m_conv_out_channels / m_group, (Dtype)1.,
				weights + m_weight_offset * g, output + m_output_offset * g,
				(Dtype)0., col_buff + m_col_offset * g);
		}
		if (!m_is_1x1) {
			ConvCol2im(col_buff, input);
		}
	}

#ifndef CPU_ONLY
	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::ForwardGpuGemm(const Dtype* input, 
					const Dtype* weights, Dtype* output, bool skip_im2col) {
		const Dtype* col_buff = input;
		if (!m_is_1x1) {
			if (!skip_im2col) {
				ConvIm2colGpu(input, m_col_buffer.GetMutableGpuData());
			}
			col_buff = m_col_buffer.GetGpuData();
		}
		for (int g = 0; g < m_group; g++) {
			JaffeGpuGemm<Dtype>(0, 0, m_conv_out_channels / m_group, 
				m_conv_out_spatial_dim, m_kernel_dim, (Dtype)1., 
				weights + m_weight_offset * g, col_buff + m_col_offset * g,
				(Dtype)1., output + m_output_offset * g);
		}
	}

	template <typename Dtype>
	void JBaseConvolutionLayer<Dtype>::ForwardGpuBias(Dtype* output, 
					const Dtype* bias) {
		JaffeGpuGemm<Dtype>(0, 0, m_num_output, m_out_spatial_dim, 1, (Dtype)1., bias, 
			m_bias_multiplier.GetGpuData(), (Dtype)1., output);
	}
#endif

    template class JBaseConvolutionLayer <float>;
    template class JBaseConvolutionLayer <double>;

} // namespace jaffe
