#include "layers/conv_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JConvolutionLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
										   const vector<JBlob<Dtype>*>& top){
		const Dtype* weight = this->m_blobs[0]->GetData();
		for(int i = 0; i < bottom.size(); i++){
			const Dtype* bottom_data = bottom[i]->GetData();
			Dtype* top_data = top[i]->GetMutableData();
			for(int n = 0; n < this->m_num; n ++){
				this->ForwardGemm(bottom_data + n * this->m_bottom_dim, weight,
					top_data + n * this->m_top_dim);
				if (this->m_bias_term){
					const Dtype* bias = this->m_blobs[1]->GetData();
					this->ForwardBias(top_data + n * this->m_top_dim, bias);
				}
			}
		}
		
		int top_0_num = top[0]->GetNum() * top[0]->GetChannels() * 
			top[0]->GetHeight() * top[0]->GetWidth();
		for (int i = 0; i < top_0_num; i ++) {
		}

	}

	template <typename Dtype>
	void JConvolutionLayer<Dtype>::ComputeOutputShape() {
		const int* kernel_shape = this->m_kernel_shape.GetData();
		const int* stride_shape = this->m_stride.GetData();
		const int* pad_shape = this->m_pad.GetData();
		const int* dilation_shape = this->m_dilation.GetData();

		this->m_output_shape.clear();

		for (int i = 0; i < this->m_num_spatial_axes; i++) {
			const int input_dim = this->GetInputShape(i + 1);
			const int kernel_extent = dilation_shape[i] * (kernel_shape[i] - 1) 
				+ 1;
			const int output_dim = (input_dim + 2 * pad_shape[i] - kernel_extent)
				/ stride_shape[i] + 1;

			this->m_output_shape.push_back(output_dim);
		}
	}

	template class JConvolutionLayer <float>;
	template class JConvolutionLayer <double>;

#ifdef CPU_ONLY
	STUB_GPU(JConvolutionLayer);
#endif

	REGISTER_LAYER_CLASS(Convolution);

} // namespace jaffe
