#ifndef BASE_CONVOLUTION_LAYER_H_H
#define BASE_CONVOLUTION_LAYER_H_H

#include "layer.h"

namespace jaffe {
	template <typename Dtype>
	class JBaseConvolutionLayer : public JLayer<Dtype>{
	public:
		JBaseConvolutionLayer(const LayerParameter& param)
			: JLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual void Reshape(const vector<JBlob<Dtype>*> & bottom,
			const vector<JBlob<Dtype>*>& top);
		virtual inline int MinBottomBlobs() const {
			return 1;
		}
		virtual inline int MinTopBlobs() const {
			return 1;
		}
		virtual inline bool EqualNumBottomTopBlobs() const {
			return true;
		}

	protected:
		void ForwardGemm(const Dtype* input, const Dtype* weigths,
			Dtype* output, bool skip_im2col = false);
		void ForwardBias(Dtype* input, const Dtype* bias);
#ifndef CPU_ONLY
		void ForwardGpuGemm(const Dtype* col_input, const Dtype* weights, 
			Dtype* output, bool skip_im2col = false);
		void ForwardGpuBias(Dtype* output, const Dtype* bias);
#endif
		void BackwardGemm(const Dtype* output, const Dtype* weights, Dtype* input);

		inline int GetInputShape(int i) {
			return (*m_bottom_shape)[m_channel_axis + i];
		} // 返回输入 Blob 尺寸第 i 个维度的值
		virtual bool ReverseDimensions() = 0; // 使用deconv时返回true
		virtual void ComputeOutputShape() = 0; // 计算输出 Blob 的尺寸


		JBlob<int> m_kernel_shape; // 滤波器核尺寸
		JBlob<int> m_stride; // stride 的尺寸
		JBlob<int> m_pad; // padding 的尺寸
		JBlob<int> m_dilation; 
		JBlob<int> m_conv_input_shape; // 卷积输入的尺寸
		vector<int> m_col_buffer_shape; // col_buffer 的尺寸
		vector<int> m_output_shape; // 输出的尺寸，由输入尺寸和相关参数计算得到
		const vector<int>* m_bottom_shape;

		int m_num_spatial_axes; // 
		int m_bottom_dim; // 输入的维度
		int m_top_dim; // 输出的维度

		int m_channel_axis;
		int m_num; // 一次性输入的图像数量
		int m_channels; // 图片通道数
		int m_group;
		int m_out_spatial_dim;
		int m_weight_offset;
		int m_num_output;
		bool m_bias_term;
		bool m_is_1x1;
		bool m_force_nd_im2col;

	private:
		void ConvIm2col(const Dtype* data, Dtype* col_buff);
		void ConvCol2im(const Dtype* col_buff, Dtype* data);
#ifndef CPU_ONLY
		void ConvIm2colGpu(const Dtype* data, Dtype* col_buff);
#endif

		int m_num_kernels_im2col;
		int m_num_kernels_col2im;
		int m_conv_out_channels;
		int m_conv_in_channels;
		int m_conv_out_spatial_dim;
		int m_kernel_dim;
		int m_col_offset;
		int m_output_offset;

		JBlob<Dtype> m_col_buffer; // 输入数据缓存区
		JBlob<Dtype> m_bias_multiplier;
	}; // class JBaseConvolutionLayer	
} // namespace jaffe
#endif
