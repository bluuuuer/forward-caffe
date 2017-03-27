#include <vector>

#include "filler.h"
#include "layers/embed_layer.h"

namespace jaffe {
    template <typename Dtype>
    void JEmbedLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
                                        const vector<JBlob<Dtype>*>& top)
    {
        m_N = this->m_layer_param.embed_param().num_output();
        //CHECK_GT(m_N, 0) << "EmbedLayer num_output must be positive.";
        m_K = this->m_layer_param.embed_param().input_dim();
        //CHECK_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
        m_bias_term = this->m_layer_param.embed_param().bias_term();
        // Check if we need to set up the weights
        if (this->m_blobs.size() > 0) {
            //LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (m_bias_term) {
                this->m_blobs.resize(2);
            } else {
                this->m_blobs.resize(1);
            }
// Initialize the weights --
// transposed from InnerProductLayer for spatial locality.
            vector<int> weight_shape(2);
            weight_shape[0] = m_K;
            weight_shape[1] = m_N;
            this->m_blobs[0].reset(new JBlob<Dtype>(weight_shape));
// fill the weights
            shared_ptr<JFiller<Dtype> > weight_filler(GetFiller<Dtype>());
            weight_filler->Fill(this->m_blobs[0].get());
// If necessary, initialize and fill the bias term
            if (m_bias_term) {
                vector<int> bias_shape(1, m_N);
                this->m_blobs[1].reset(new JBlob<Dtype>(bias_shape));
                shared_ptr<JFiller<Dtype> > bias_filler(GetFiller<Dtype>());
                bias_filler->Fill(this->m_blobs[1].get());
            }
        }  // parameter initialization
        this->m_param_propagate_down.resize(this->m_blobs.size(), false);
    }


    template <typename Dtype>
    void JEmbedLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        // Figure out the dimensions
        m_M = bottom[0]->GetCount();
        vector<int> top_shape = bottom[0]->GetShape();
        top_shape.push_back(m_N);
        top[0]->Reshape(top_shape);
        // Set up the bias multiplier
        if (m_bias_term) {
            vector<int> bias_shape(1, m_M);
            m_bias_multiplier.Reshape(bias_shape);
            JaffeSet(m_M, Dtype(1), m_bias_multiplier.GetMutableData());
        }
    }

    template <typename Dtype>
    void JEmbedLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        const Dtype* weight = this->m_blobs[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        int index;
        for (int n = 0; n < m_M; ++n) {
            index = static_cast<int>(bottom_data[n]);
        //DCHECK_GE(index, 0);
        //DCHECK_LT(index, m_K);
        //DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n]) << "non-integer input";
            JaffeCopy(m_N, weight + index * m_N, top_data + n * m_N);
        }
        if (m_bias_term) {
            const Dtype* bias = this->m_blobs[1]->GetData();
            JaffeGemm<Dtype>(0, 0, m_M, m_N, 1, Dtype(1), m_bias_multiplier.GetData(), bias, Dtype(1), top_data);
        }
    }
    INSTANTIATE_CLASS(JEmbedLayer);
    REGISTER_LAYER_CLASS(Embed);
}
