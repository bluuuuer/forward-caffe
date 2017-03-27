#include "layers/prelu_layer.h"

namespace jaffe {

	template <typename Dtype>
	void JPReLULayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
    		const vector<JBlob<Dtype>*>& top) {
		if (bottom[0]->GetNumAxes() < 2) {
			cout << "Error: Number of axes of bottom blob must be >= 2." << endl;
		}
  		PReLUParameter prelu_param = this->GetLayerParam().prelu_param();
  		int channels = bottom[0]->GetChannels();
  		m_channel_shared = prelu_param.channel_shared();
  		if (this->m_blobs.size() > 0) {
    		cout << "Skipping parameter initialization" << endl;
  		} else {
    		this->m_blobs.resize(1);
    		if (m_channel_shared) {
      			this->m_blobs[0].reset(new JBlob<Dtype>(vector<int>(0)));
    		} else {
      		this->m_blobs[0].reset(new JBlob<Dtype>(vector<int>(1, channels)));
    		}
			//=======改为constant填充
    		shared_ptr<JFiller<Dtype> > filler(GetFiller<Dtype>());
    		filler->Fill(this->m_blobs[0].get());
  		}
  		if (m_channel_shared) {
			if (this->m_blobs[0]->GetCount() != 1) {
				cout << "Error: Negative slope size is inconsistent with prototxt"
					<< " config." << endl;
			}
  		} else {
			if (this->m_blobs[0]->GetCount() != channels) {
				cout << "Error: Negative slope size is inconsistent with prototxt "
					<< "config." << endl;
			}
  		}

  // Propagate gradients to the parameters (as directed by backward pass).
		//==========Forward 中没有用到
  		//this->m_param_propagate_down.resize(this->m_blobs.size(), true);
  		//m_multiplier.Reshape(vector<int>(1, bottom[0]->GetCount(1)));
  		//m_backward_buff.Reshape(vector<int>(1, bottom[0]->GetCount(1)));
  		//JaffeSet(m_multiplier.GetCount(), Dtype(1), m_multiplier.GetMutableData());
	}

	template <typename Dtype>
	void JPReLULayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
    		const vector<JBlob<Dtype>*>& top) {
		if (bottom[0]->GetNumAxes() < 2) {
      		cout << "Error: Number of axes of bottom blob must be >=2." << endl;
		}
  		top[0]->ReshapeLike(*bottom[0]);
  		if (bottom[0] == top[0]) {
    	// For in-place computation
    		m_bottom_memory.ReshapeLike(*bottom[0]);
  		}
	}

	template <typename Dtype>
	void JPReLULayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom,
    		const vector<JBlob<Dtype>*>& top) {
  		const Dtype* bottom_data = bottom[0]->GetData();
  		Dtype* top_data = top[0]->GetMutableData();

  		const int count = bottom[0]->GetCount();
  		const int dim = bottom[0]->GetCount(2);
  		const int channels = bottom[0]->GetChannels();
		
  		const Dtype* slope_data = this->m_blobs[0]->GetData();

  		// For in-place computation
  		if (bottom[0] == top[0]) {
    		JaffeCopy(count, bottom_data, m_bottom_memory.GetMutableData());
  		}

  		// if channel_shared, channel index in the following computation becomes
  		// always zero.
  		const int div_factor = m_channel_shared ? channels : 1;
  		for (int i = 0; i < count; ++i) {
    		int c = (i / dim) % channels / div_factor;
    		top_data[i] = std::max(bottom_data[i], Dtype(0)) + slope_data[c] * 
				std::min(bottom_data[i], Dtype(0));
  		}
	}

	template class JPReLULayer<float>;
	template class JPReLULayer<double>;

#ifdef CPU_ONLY
	STUB_GPU(JPReLULayer);
#endif

	REGISTER_LAYER_CLASS(PReLU);

}  // namespace jaffe
