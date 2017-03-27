//
// Created by huangshize on 16-5-13.
//

#ifndef JAFFE_BIAS_LAYER_H
#define JAFFE_BIAS_LAYER_H

#include "layer.h"

namespace jaffe {
    // 计算两个 input blobs 的和， 其中后面的 blob 的尺寸匹配前面的 blob 的尺寸

    template<typename Dtype>
    class BiasLayer : public JLayer<Dtype>{
    public:
        explicit BiasLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
                                const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
                             const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Bias"; }
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MaxBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

        virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
                             const vector<JBlob<Dtype>*>& top);

    private:
        JBlob<Dtype> m_bias_multiplier;
        int m_outer_dim;
        int m_bias_dim;
        int m_inner_dim;
        int m_dim;

    };
}

#endif //JAFFE_BIAS_LAYER_H
