#include <iostream>

#include "layer.h"

namespace jaffe {

	template <typename Dtype>
	Dtype JLayer<Dtype>::LayerForward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) {
		Dtype loss = 0;
		Reshape(bottom, top);
		switch(Jaffe::GetMode()) {
		case Jaffe::CPU:
			Forward(bottom, top);
			for (int top_id = 0; top_id < top.size(); top_id++) {
				if (!this->GetLoss(top_id))
					continue;
				const int count = top[top_id]->GetCount();
				const Dtype* data = top[top_id]->GetData();
				const Dtype* loss_weights = top[top_id]->GetDiff();
				loss += JaffeDot(count, data, loss_weights);
			}
			break;
		case Jaffe::GPU:
			ForwardGpu(bottom, top);
#ifndef CPU_ONLY
			for (int top_id = 0; top_id < top.size(); top_id ++) {
				if (!this->GetLoss(top_id)) {
					continue;
				}
				const int count = top[top_id]->GetCount();
				const Dtype* data = top[top_id]->GetGpuData();
				const Dtype* loss_weights = top[top_id]->GetGpuDiff();
				Dtype blob_loss = 0;
				JaffeGpuDot(count, data, loss_weights, &blob_loss);
				loss += blob_loss;
			}
#endif
			break;
		default:
			cout << __FILE__ << "\t" << __LINE__ << "\t Unknown jaffe mode." << endl;
		}
		return loss;
	}

	template class JLayer <float>;
	template class JLayer <double>;

} // namespace jaffe
