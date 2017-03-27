#include "layers/deconv_layer.h"
#include "util/io.h"

namespace jaffe {

	template <typename Dtype>
	void JDeconvolutionLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
										   const vector<JBlob<Dtype>*>& top){
		//cout << "JDeconvolutionLayer::Forward()" << endl;
		/*int bottom_0_num = bottom[0]->GetNum() * bottom[0]->GetChannels() * 
			bottom[0]->GetHeight() * bottom[0]->GetWidth();
		for (int i = 0; i < bottom_0_num; i ++) {
			cout << "bottom[0] " << i << "): " << bottom[0]->GetData()[i] << endl;
		}*/
		//cout << "Forward()" << endl;
		const Dtype* weight = this->m_blobs[0]->GetData();
		for(int i = 0; i < bottom.size(); i++){
			const Dtype* bottom_data = bottom[i]->GetData();
			Dtype* top_data = top[i]->GetMutableData();
			for(int n = 0; n < this->m_num; n ++){
				this->BackwardGemm(bottom_data + n * this->m_bottom_dim, weight,
					top_data + n * this->m_top_dim);
				if (this->m_bias_term){
					const Dtype* bias = this->m_blobs[1]->GetData();
					this->ForwardBias(top_data + n * this->m_top_dim, bias);
				}
			}
		}
		/*int top_0_num = top[0]->GetCount();
		for (int i = 0; i < top_0_num; i ++) {
			cout << "top[0] " << i << "): " << top[0]->GetData()[i] << endl;
		}*/

	}

	template <typename Dtype>
	void JDeconvolutionLayer<Dtype>::ComputeOutputShape() {
		const int* kernel_shape = this->m_kernel_shape.GetData();
		const int* stride_shape = this->m_stride.GetData();
		const int* pad_shape = this->m_pad.GetData();
		const int* dilation_shape = this->m_dilation.GetData();

		this->m_output_shape.clear();

		for (int i = 0; i < this->m_num_spatial_axes; i++) {
			const int input_dim = this->GetInputShape(i + 1);
			const int kernel_extent = dilation_shape[i] * (kernel_shape[i] - 1) + 1;
			const int output_dim = stride_shape[i] * (input_dim - 1) + kernel_extent 
				- 2 * pad_shape[i];
			this->m_output_shape.push_back(output_dim);
		}
	}

	template class JDeconvolutionLayer <float>;
	template class JDeconvolutionLayer <double>;

	REGISTER_LAYER_CLASS(Deconvolution);

} // namespace jaffe
