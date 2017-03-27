// Unfinished
#include "layers/mvn_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JMVNLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        top[0]->Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), bottom[0]->GetHeight(), bottom[0]->GetWidth());
        m_mean.Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), 1, 1);
        m_variance.Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), 1, 1);
        m_temp.Reshape(bottom[0]->GetNum(), bottom[0]->GetChannels(), bottom[0]->GetHeight(), bottom[0]->GetWidth());
        if ( this->m_layer_param.mvn_param().across_channels() ) {
            m_sum_multiplier.Reshape(1, bottom[0]->GetChannels(), bottom[0]->GetHeight(), bottom[0]->GetWidth());
        } else {
            m_sum_multiplier.Reshape(1, 1, bottom[0]->GetHeight(), bottom[0]->GetWidth());
        }
        Dtype* multiplier_data = m_sum_multiplier.GetMutableData();
        JaffeSet(m_sum_multiplier.GetCount(), Dtype(1), multiplier_data);
        m_eps = this->m_layer_param.mvn_param().eps();
    }

    template <typename Dtype>
    void JMVNLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        int num;
        if (this->m_layer_param.mvn_param().across_channels())
            num = bottom[0]->GetNum();
        else
            num = bottom[0]->GetNum() * bottom[0]->GetChannels();

        int dim = bottom[0]->GetCount() / num;

// subtract mean
        JaffeGemv<Dtype>(0, num, dim, 1. / dim, bottom_data, m_sum_multiplier.GetData(), 0., m_mean.GetMutableData());  // EX
        JaffeGemm<Dtype>(0, 0, num, dim, 1, -1., m_mean.GetData(), m_sum_multiplier.GetData(), 0., m_temp.GetMutableData());
        //caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

        if (this->m_layer_param.mvn_param().normalize_variance()) {
// compute variance using var(X) = E((X-EX)^2)
            JaffePowx(bottom[0]->GetCount(), top_data, Dtype(2), m_temp.GetMutableData());  // (X-EX)^2
            JaffeGemv<Dtype>(0, num, dim, 1. / dim, m_temp.GetData(), m_sum_multiplier.GetData(), 0., m_variance.GetMutableData());
// normalize variance
            JaffePowx(m_variance.GetCount(), m_variance.GetData(), Dtype(0.5), m_variance.GetMutableData());

            JaffeAddScalar(m_variance.GetCount(), m_eps, m_variance.GetMutableData());

            JaffeGemm<Dtype>(0, 0, num, dim, 1, 1., m_variance.GetData(), m_sum_multiplier.GetData(), 0., m_temp.GetMutableData());

            JaffeDiv(m_temp.GetCount(), top_data, m_temp.GetData(), top_data);
        }
    }
    INSTANTIATE_CLASS(JMVNLayer);
    REGISTER_LAYER_CLASS(MVN);
}
