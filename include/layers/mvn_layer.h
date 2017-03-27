#ifndef JAFFE_MVN_LAYER_H
#define JAFFE_MVN_LAYER_H

#include "layer.h"

namespace jaffe {
    // 将 input 每一维度减去自身均值，再除以自身标准差
    template <typename Dtype>
    class JMVNLayer : public JLayer<Dtype>
    {
    public:
        explicit JMVNLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "MVN"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        JBlob<Dtype> m_mean, m_variance, m_temp;

        /// sum_multiplier is used to carry out sum using BLAS
        JBlob<Dtype> m_sum_multiplier;
        Dtype m_eps;
    };
}

#endif //JAFFE_MVN_LAYER_H
