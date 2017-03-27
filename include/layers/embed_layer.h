//
// Created by huangshize on 16-5-18.
//

#ifndef JAFFE_EMBED_LAYER_H
#define JAFFE_EMBED_LAYER_H

#include "layer.h"

namespace jaffe {

    template <typename Dtype>
    class JEmbedLayer : public JLayer<Dtype>
    {
    public:
        explicit JEmbedLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Embed"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        int m_M;
        int m_K;
        int m_N;
        bool m_bias_term;
        JBlob<Dtype> m_bias_multiplier;
    };
}
#endif //JAFFE_EMBED_LAYER_H
