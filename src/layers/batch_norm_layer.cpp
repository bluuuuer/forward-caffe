// Unfinished
#include <algorithm>

#include "layers/batch_norm_layer.h"

namespace jaffe {
    template <typename Dtype>
    void JBatchNormLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype> *> &bottom, const vector<JBlob<Dtype> *> &top)
    {
        BatchNormParameter param = this->m_layer_param.batch_norm_param();
        m_moving_average_fraction = param.moving_average_fraction();
        // m_use_global_stats = this->m_phase == TEST;
        m_use_global_stats = 1;
        if (param.has_use_global_stats())
        {
            m_use_global_stats = param.use_global_stats();
        }
        if (bottom[0]->GetNumAxes() == 1)
        {
            m_channels = 1;
        }
        else
        {
            m_channels = bottom[0]->GetShape(1);
        }
        m_eps = param.eps();
        if (this->m_blobs.size() > 0)
        {
            // 已经完成参数初始化
        }
        else
        {
            this->m_blobs.resize(3);
            vector<int> sz;
            sz.push_back(m_channels);
            this->m_blobs[0].reset(new JBlob<Dtype>(sz));
            this->m_blobs[1].reset(new JBlob<Dtype>(sz));
            sz[0] = 1;
            this->m_blobs[2].reset(new JBlob<Dtype>(sz));
            for (int i = 0; i < 3; ++i) {
                JaffeSet(this->m_blobs[i]->GetCount(), Dtype(0), this->m_blobs[i]->GetMutableData());
            }
        }
    }

    template <typename Dtype>
    void JBatchNormLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        top[0]->ReshapeLike(*bottom[0]);

        vector<int> sz;
        sz.push_back(m_channels);
        m_mean.Reshape(sz);
        m_variance.Reshape(sz);
        m_temp.ReshapeLike(*bottom[0]);
        m_x_norm.ReshapeLike(*bottom[0]);
        sz[0] = bottom[0]->GetShape(0);
        m_batch_sum_multiplier.Reshape(sz);

        int spatial_dim = bottom[0]->GetCount() / (m_channels * bottom[0]->GetShape(0));
        if (m_spatial_sum_multiplier.GetNumAxes() == 0 || m_spatial_sum_multiplier.GetShape(0) != spatial_dim)
        {
            sz[0] = spatial_dim;
            m_spatial_sum_multiplier.Reshape(sz);
            Dtype* multiplier_data = m_spatial_sum_multiplier.GetMutableData();
            JaffeSet(m_spatial_sum_multiplier.GetCount(), Dtype(1), multiplier_data);
        }

        int numbychans = m_channels * bottom[0]->GetShape(0);
        if (m_num_by_chans.GetNumAxes() == 0 ||
            m_num_by_chans.GetShape(0) != numbychans) {
            sz[0] = numbychans;
            m_num_by_chans.Reshape(sz);
            JaffeSet(m_batch_sum_multiplier.GetCount(), Dtype(1),
                     m_batch_sum_multiplier.GetMutableData());
        }
    }

    template <typename Dtype>
    void JBatchNormLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        int num = bottom[0]->GetShape(0);
        int spatial_dim = bottom[0]->GetCount() / (bottom[0]->GetShape(0)*m_channels);

        if (bottom[0] != top[0]) {
            JaffeCopy(bottom[0]->GetCount(), bottom_data, top_data);
        }

        if (m_use_global_stats)
        {
            const Dtype scale_factor = this->m_blobs[2]->GetData()[0] == 0 ?
                                       0 : 1 / this->m_blobs[2]->GetData()[0];
            // Unfinished
//            JaffeScal(m_variance.GetCount(), scale_factor,
//                            this->m_blobs[0]->GetData(), m_mean.GetMutableData());
//            JaffeScal(m_variance.GetCount(), scale_factor,
//                            this->m_blobs[1]->GetData(), m_variance.GetMutableData());
        }
        else
        {
            JaffeGemv<Dtype>(0, m_channels * num, spatial_dim,
                                  1. / (num * spatial_dim), bottom_data,
                                  m_spatial_sum_multiplier.GetData(), 0.,
                                  m_num_by_chans.GetMutableData());
            JaffeGemv<Dtype>(1, num, m_channels, 1.,
                                  m_num_by_chans.GetData(), m_batch_sum_multiplier.GetData(), 0.,
                                  m_mean.GetMutableData());
        }

        JaffeGemm<Dtype>(0, 0, num, m_channels, 1, 1,
                              m_batch_sum_multiplier.GetData(), m_mean.GetData(), 0.,
                              m_num_by_chans.GetMutableData());
        JaffeGemm<Dtype>(0, 0, m_channels * num,
                              spatial_dim, 1, -1, m_num_by_chans.GetData(),
                              m_spatial_sum_multiplier.GetData(), 1., top_data);

        if (!m_use_global_stats) {
            // 计算 variance var(X) = E((X-EX)^2)
            JaffePowx(top[0]->GetCount(), top_data, Dtype(2),
                       m_temp.GetMutableData());  // (X-EX)^2
            JaffeGemv<Dtype>(0, m_channels * num, spatial_dim,
                                  1. / (num * spatial_dim), m_temp.GetData(),
                                  m_spatial_sum_multiplier.GetData(), 0.,
                                  m_num_by_chans.GetMutableData());
            JaffeGemv<Dtype>(1, num, m_channels, 1.,
                                  m_num_by_chans.GetData(), m_batch_sum_multiplier.GetData(), 0.,
                                  m_variance.GetMutableData());  // E((X_EX)^2)

            this->m_blobs[2]->GetMutableData()[0] *= m_moving_average_fraction;
            this->m_blobs[2]->GetMutableData()[0] += 1;
            // Unfinished
//            JaffeAxpby(m_mean.GetCount(), Dtype(1), m_mean.GetData(),
//                            m_moving_average_fraction, this->m_blobs[0]->GetMutableData());
            int m = bottom[0]->GetCount()/m_channels;
            Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
//            JaffeAxpby(m_variance.GetCount(), bias_correction_factor,
//                            m_variance.GetData(), m_moving_average_fraction,
//                            this->m_blobs[1]->GetMutableData());
        }

        JaffeAddScalar(m_variance.GetCount(), m_eps, m_variance.GetMutableData());
        JaffePowx(m_variance.GetCount(), m_variance.GetData(), Dtype(0.5),
                   m_variance.GetMutableData());

        JaffeGemm<Dtype>(0, 0, num, m_channels, 1, 1,
                              m_batch_sum_multiplier.GetData(), m_variance.GetData(), 0.,
                              m_num_by_chans.GetMutableData());
        JaffeGemm<Dtype>(0, 0, m_channels * num,
                              spatial_dim, 1, 1., m_num_by_chans.GetData(),
                              m_spatial_sum_multiplier.GetData(), 0., m_temp.GetMutableData());
        JaffeDiv(m_temp.GetCount(), top_data, m_temp.GetData(), top_data);

        JaffeCopy(m_x_norm.GetCount(), top_data, m_x_norm.GetMutableData());
    }
    INSTANTIATE_CLASS(JBatchNormLayer);
    REGISTER_LAYER_CLASS(BatchNorm);
}
