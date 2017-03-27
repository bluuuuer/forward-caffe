#include "layers/pooling_layer.h"

namespace jaffe {

	template <typename Dtype>
	__global__ void MaxPoolForward(const int nthreads, const Dtype* const bottom_data, 
		const int num, const int channels, const int height, const int width, 
		const int pooled_height, const int pooled_width, const int kernel_h, 
		const int kernel_w, const int stride_h, const int stride_w, const int pad_h, 
		const int pad_w, Dtype* const top_data, int* mask, Dtype* top_mask) {

  		CUDA_KERNEL_LOOP(index, nthreads) {
    		const int pw = index % pooled_width;
    		const int ph = (index / pooled_width) % pooled_height;
    		const int c = (index / pooled_width / pooled_height) % channels;
    		const int n = index / pooled_width / pooled_height / channels;
    		int hstart = ph * stride_h - pad_h;
    		int wstart = pw * stride_w - pad_w;
    		const int hend = min(hstart + kernel_h, height);
    		const int wend = min(wstart + kernel_w, width);
    		hstart = max(hstart, 0);
    		wstart = max(wstart, 0);
    		Dtype maxval = -FLT_MAX;
    		int maxidx = -1;
    		const Dtype* const bottom_slice = bottom_data + (n * channels + c) * 
				height * width;
    		for (int h = hstart; h < hend; ++h) {
      			for (int w = wstart; w < wend; ++w) {
        			if (bottom_slice[h * width + w] > maxval) {
          				maxidx = h * width + w;
          				maxval = bottom_slice[maxidx];
        			}
      			}
    		}
    		top_data[index] = maxval;
    		if (mask) {
      			mask[index] = maxidx;
    		} else {
      			top_mask[index] = maxidx;
    		}
  		}
	}

	template <typename Dtype>
	__global__ void AvePoolForward(const int nthreads, const Dtype* const bottom_data, 
		const int num, const int channels, const int height, const int width, 
		const int pooled_height, const int pooled_width, const int kernel_h, 
		const int kernel_w, const int stride_h, const int stride_w, const int pad_h, 
		const int pad_w, Dtype* const top_data) {
			
  		CUDA_KERNEL_LOOP(index, nthreads) {
    		const int pw = index % pooled_width;
    		const int ph = (index / pooled_width) % pooled_height;
    		const int c = (index / pooled_width / pooled_height) % channels;
    		const int n = index / pooled_width / pooled_height / channels;
    		int hstart = ph * stride_h - pad_h;
    		int wstart = pw * stride_w - pad_w;
    		int hend = min(hstart + kernel_h, height + pad_h);
    		int wend = min(wstart + kernel_w, width + pad_w);
    		const int pool_size = (hend - hstart) * (wend - wstart);
    		hstart = max(hstart, 0);
    		wstart = max(wstart, 0);
    		hend = min(hend, height);
    		wend = min(wend, width);
    		Dtype aveval = 0;
    		const Dtype* const bottom_slice = bottom_data + (n * channels + c) * 
				height * width;
    		for (int h = hstart; h < hend; ++h) {
      			for (int w = wstart; w < wend; ++w) {
        			aveval += bottom_slice[h * width + w];
      			}
    		}
    		top_data[index] = aveval / pool_size;
  		}
	}


	template <typename Dtype>
	__global__ void StoPoolForward(const int nthreads, 
		const Dtype* const bottom_data, const int num, const int channels, 
		const int height, const int width, const int pooled_height, 
		const int pooled_width, const int kernel_h, const int kernel_w, 
		const int stride_h, const int stride_w, Dtype* const top_data) {
  			
		CUDA_KERNEL_LOOP(index, nthreads) {
    		const int pw = index % pooled_width;
    		const int ph = (index / pooled_width) % pooled_height;
    		const int c = (index / pooled_width / pooled_height) % channels;
    		const int n = index / pooled_width / pooled_height / channels;
    		const int hstart = ph * stride_h;
    		const int hend = min(hstart + kernel_h, height);
    		const int wstart = pw * stride_w;
    		const int wend = min(wstart + kernel_w, width);
    		// We set cumsum to be 0 to avoid divide-by-zero problems
    		Dtype cumsum = FLT_MIN;
    		Dtype cumvalues = 0.;
    		const Dtype* const bottom_slice = bottom_data + (n * channels + c) 
				* height * width;
    		// First pass: get sum
    		for (int h = hstart; h < hend; ++h) {
      			for (int w = wstart; w < wend; ++w) {
        			cumsum += bottom_slice[h * width + w];
        			cumvalues += bottom_slice[h * width + w] * 
						bottom_slice[h * width + w];
      			}
    		}
    		top_data[index] = cumvalues / cumsum;
  		}
	}


	template <typename Dtype>
	void JPoolingLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
      	const vector<JBlob<Dtype>*>& top) {
			
  		const Dtype* bottom_data = bottom[0]->GetGpuData();
  		Dtype* top_data = top[0]->GetMutableGpuData();
  		int count = top[0]->GetCount();
  		// We'll output the mask to top[1] if it's of size >1.
  		const bool use_top_mask = top.size() > 1;
  		int* mask = NULL;
  		Dtype* top_mask = NULL;
  		switch (this->m_layer_param.pooling_param().pool()) {
  		case PoolingParameter_PoolMethod_MAX:
    		if (use_top_mask) {
      			top_mask = top[1]->GetMutableGpuData();
    		} else {
      			mask = m_max_idx.GetMutableGpuData();
    		}
    		MaxPoolForward<Dtype>
				<<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(count, 
				bottom_data, bottom[0]->GetNum(), m_channels, m_height, m_width, 
				m_pooled_height, m_pooled_width, m_kernel_h, m_kernel_w, m_stride_h, 
				m_stride_w, m_pad_h, m_pad_w, top_data, mask, top_mask);
    		break;
  		case PoolingParameter_PoolMethod_AVE:
    		AvePoolForward<Dtype>
				<<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(count, 
				bottom_data, bottom[0]->GetNum(), m_channels, m_height, m_width, 
				m_pooled_height, m_pooled_width, m_kernel_h, m_kernel_w, m_stride_h, 
				m_stride_w, m_pad_h, m_pad_w, top_data);
    		break;
  		case PoolingParameter_PoolMethod_STOCHASTIC:
      		StoPoolForward<Dtype>
				<<<JAFFE_GET_BLOCKS(count), JAFFE_CUDA_NUM_THREADS>>>(count, 
				bottom_data, bottom[0]->GetNum(), m_channels, m_height, m_width, 
				m_pooled_height, m_pooled_width, m_kernel_h, m_kernel_w, m_stride_h, 
				m_stride_w, top_data);
    		break;
  		default:
			cout << __FILE__ << "\t" << __LINE__  << "\tERROR: Unknown pooling "
				<< "method." << endl;
  		}
  		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FORWARD(JPoolingLayer); 
}  // namespace jaffe
