#include "layers/concat_layer.h"

namespace jaffe {

	template <typename Dtype>
	__global__ void ConcatKernel(const int nthreads, const Dtype* in_data, 
		const int num_concats, const int concat_size, const int top_concat_axis, 
		const int bottom_concat_axis, const int offset_concat_axis, Dtype* out_data) {
  	
		CUDA_KERNEL_LOOP(index, nthreads) {
    		const int total_concat_size = concat_size * bottom_concat_axis;
    		const int concat_num = index / total_concat_size;
    		const int concat_index = index % total_concat_size;
    		const int top_index = concat_index +
        		(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
      		out_data[top_index] = in_data[index];
  		}
	}

	template <typename Dtype>
	void JConcatLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
      	const vector<JBlob<Dtype>*>& top) {
  
		if (bottom.size() == 1) { 
			return; 
		}
  		Dtype* top_data = top[0]->GetMutableGpuData();
  		int offset_concat_axis = 0;
  		const int top_concat_axis = top[0]->GetShape(m_concat_axis);
  		for (int i = 0; i < bottom.size(); ++i) {
    		const Dtype* bottom_data = bottom[i]->GetGpuData();
    		const int bottom_concat_axis = bottom[i]->GetShape(m_concat_axis);
    		const int bottom_concat_size = bottom_concat_axis * m_concat_input_size;
    		const int nthreads = bottom_concat_size * m_num_concats;
    		ConcatKernel<Dtype>  
        		<<<JAFFE_GET_BLOCKS(nthreads), JAFFE_CUDA_NUM_THREADS>>>(
        		nthreads, bottom_data, m_num_concats, m_concat_input_size,
        		top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    		offset_concat_axis += bottom_concat_axis;
  		}
	}

INSTANTIATE_LAYER_GPU_FORWARD(JConcatLayer);

}  // namespace jaffe
