//
// Created by huangshize on 16-5-16.
//

#ifndef JAFFE_BATCH_NORM_LAYER_HPP_H
#define JAFFE_BATCH_NORM_LAYER_HPP_H

#include "layer.h"

namespace jaffe {

    // 该layer主要的方法是 Batch Normalization
    // 在该 batch 内将 input 每一维度减去自身均值，再除以自身标准差
    // Normalizes the input to have 0-mean and/or unit (1) variance across the batch.
    // BLOG: http://blog.csdn.net/happynear/article/details/44238541

    template <typename Dtype>
    class JBatchNormLayer : public JLayer<Dtype>
    {
    public:
        explicit JBatchNormLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "BatchNorm"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        JBlob<Dtype> m_mean, m_variance, m_temp, m_x_norm;
        bool m_use_global_stats;
        Dtype m_moving_average_fraction;
        int m_channels;
        Dtype m_eps;

        JBlob<Dtype> m_batch_sum_multiplier;
        JBlob<Dtype> m_num_by_chans;
        JBlob<Dtype> m_spatial_sum_multiplier;
    };


}
#endif //JAFFE_BATCH_NORM_LAYER_HPP_H
