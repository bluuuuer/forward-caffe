#include "layers/lrn_layer.h"

namespace jaffe{ 

	template <typename Dtype>
	__global__ void LRNFillScale(const int nthreads, const Dtype* const in, 
		const int num, const int channels, const int height, const int width, 
		const int size, const Dtype alpha_over_size, const Dtype k, 
		Dtype* const scale) {

  		CUDA_KERNEL_LOOP(index, nthreads) {
    		const int w = index % width;
    		const int h = (index / width) % height;
    		const int n = index / width / height;
    		const int offset = (n * channels * height + h) * width + w;
    		const int step = height * width;
    		const Dtype* const in_off = in + offset;
    		Dtype* const scale_off = scale + offset;
    		int head = 0;
    		const int pre_pad = (size - 1) / 2;
    		const int post_pad = size - pre_pad - 1;
    		Dtype accum_scale = 0;
    		// fill the scale at [n, :, h, w]
    		// accumulate values
    		while (head < post_pad && head < channels) {
      			accum_scale += in_off[head * step] * in_off[head * step];
      			++head;
    		}
    		// both add and subtract
    		while (head < channels) {
      			accum_scale += in_off[head * step] * in_off[head * step];
      			if (head - size >= 0) {
        			accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      			}
      			scale_off[(head - post_pad) * step] = k + accum_scale * 
					alpha_over_size;
      			++head;
    		}
    		// subtract only
    		while (head < channels + post_pad) {
      			if (head - size >= 0) {
        			accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      			}
      			scale_off[(head - post_pad) * step] = k + accum_scale * 
					alpha_over_size;
      			++head;
    		}
  		}
	}

	template <typename Dtype>
	__global__ void LRNComputeOutput(const int nthreads, const Dtype* const in,
		const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			out[index] = in[index] * pow(scale[index], negative_beta);
		}
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		switch (this->m_layer_param.lrn_param().norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			CrossChannelForwardGpu(bottom, top);
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
		  	WithinChannelForward(bottom, top);
			break;
		default:
			cout << __FILE__ << "\t" << __LINE__ << "\tERROR: Unknown normalization" 
				<< " region." << endl;	
		}
	}

	template <typename Dtype>
	void JLRNLayer<Dtype>::CrossChannelForwardGpu(const vector<JBlob<Dtype>*>& bottom,
					const vector<JBlob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->GetGpuData();
		Dtype* top_data = top[0]->GetMutableGpuData();
		Dtype* scale_data = m_scale.GetMutableGpuData();
		int n_threads = m_num * m_height * m_width;
		LRNFillScale<<<JAFFE_GET_BLOCKS(n_threads), JAFFE_CUDA_NUM_THREADS>>>(
			n_threads, bottom_data, m_num, m_channels, m_height, m_width, m_size,
			m_alpha / m_size, m_k, scale_data);
		CUDA_POST_KERNEL_CHECK;
		n_threads = bottom[0]->GetCount();
		LRNComputeOutput<<<JAFFE_GET_BLOCKS(n_threads), JAFFE_CUDA_NUM_THREADS>>>(
			n_threads, bottom_data, scale_data, -m_beta, top_data);
		CUDA_POST_KERNEL_CHECK;
	}
	template void JLRNLayer<float>::CrossChannelForwardGpu(
		const vector<JBlob<float>*>& bottom, const vector<JBlob<float>*>& top);
	template void JLRNLayer<double>::CrossChannelForwardGpu(
		const vector<JBlob<double>*>& bottom, const vector<JBlob<double>*>& top);

	INSTANTIATE_LAYER_GPU_FORWARD(JLRNLayer);
	
} // namespace jaffe
