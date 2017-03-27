#include "layers/silence_layer.h"

namespace jaffe {

	template class JSilenceLayer<float>;
	template class JSilenceLayer<double>;

#ifdef CPU_ONLY
	STUB_GPU(JSilenceLayer);
#endif

	REGISTER_LAYER_CLASS(Silence);

} // namespace jaffe
