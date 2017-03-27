#ifndef JAFFE_SPP_LAYER_H
#define JAFFE_SPP_LAYER_H

#include <memory>

#include "layers/flatten_layer.h"
#include "layers/split_layer.h"
#include "layers/concat_layer.h"
#include "layers/pooling_layer.h"

namespace jaffe {
    // spatial pyramid pooling
    // pooling 的一种，使得输入的图像可以多尺寸化
    // 把任何尺度的图像的卷积特征转化成相同维度
    template <typename Dtype>
    class JSPPLayer : public JLayer<Dtype>
    {
    public:
        explicit JSPPLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "SPP"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        // calculates the kernel and stride dimensions for the pooling layer,
        // returns a correctly configured LayerParameter for a PoolingLayer
        virtual LayerParameter GetPoolingParam(const int pyramid_level,
                                               const int bottom_h,
                                               const int bottom_w,
                                               const SPPParameter spp_param);

        int m_pyramid_height;
        int m_bottom_h, m_bottom_w;
        int m_num;
        int m_channels;
        int m_kernel_h, m_kernel_w;
        int m_pad_h, m_pad_w;
        bool m_reshaped_first_time;

        /// the internal Split layer that feeds the pooling layers
        shared_ptr<JSplitLayer<Dtype> > m_split_layer;
        /// top vector holder used in call to the underlying SplitLayer::Forward
        vector<JBlob<Dtype>*> m_split_top_vec;
        /// bottom vector holder used in call to the underlying PoolingLayer::Forward
        vector<vector<JBlob<Dtype>*>*> m_pooling_bottom_vecs;
        /// the internal Pooling layers of different kernel sizes
        vector<shared_ptr<JPoolingLayer<Dtype> > > m_pooling_layers;
        /// top vector holders used in call to the underlying PoolingLayer::Forward
        vector<vector<JBlob<Dtype>*>*> m_pooling_top_vecs;
        /// pooling_outputs stores the outputs of the PoolingLayers
        vector<JBlob<Dtype>*> m_pooling_outputs;
        /// the internal Flatten layers that the Pooling layers feed into
        vector<JFlattenLayer<Dtype>*> m_flatten_layers;
        /// top vector holders used in call to the underlying FlattenLayer::Forward
        vector<vector<JBlob<Dtype>*>*> m_flatten_top_vecs;
        /// flatten_outputs stores the outputs of the FlattenLayers
        vector<JBlob<Dtype>*> m_flatten_outputs;
        /// bottom vector holder used in call to the underlying ConcatLayer::Forward
        vector<JBlob<Dtype>*> m_concat_bottom_vec;
        /// the internal Concat layers that the Flatten layers feed into
        shared_ptr<JConcatLayer<Dtype> > m_concat_layer;
    };
}
#endif //JAFFE_SPP_LAYER_H
