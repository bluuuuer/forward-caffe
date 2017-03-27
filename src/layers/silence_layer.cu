#include "layers/silence_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JSilenceLayer<Dtype>::ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
      	const vector<JBlob<Dtype>*>& top) {
  		// Do nothing.
	}

INSTANTIATE_LAYER_GPU_FORWARD(JSilenceLayer);

}  // namespace jaffe
