//
// Created by huangshize on 16-5-19.
//

#ifndef JAFFE_FILTER_LAYER_H
#define JAFFE_FILTER_LAYER_H

#include "layer.h"

namespace jaffe {
    /**
 * @brief Takes two+ Blobs, interprets last Blob as a selector and
 *  filter remaining Blobs accordingly with selector data (0 means that
 * the corresponding item has to be filtered, non-zero means that corresponding
 * item needs to stay).
 */
    // 选用 bottom 中 最后一个作为 selector 来 filter 剩下的所有 blob (前 n-1 个)
    // 最后一个 bottom 的 blob 中 0 表示当前数据需要被filter ，非0 表示但前数据保留
    template <typename Dtype>
    class JFilterLayer : public JLayer<Dtype>
    {
    public:
        explicit JFilterLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Filter"; }
        // bottom 比 top 多一个
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        bool m_first_reshape;
        vector<int> m_indices_to_forward;
    };
}
#endif //JAFFE_FILTER_LAYER_H
