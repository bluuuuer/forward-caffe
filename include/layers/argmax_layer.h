//
// Created by huangshize on 16-5-13.
//

#ifndef JAFFE_ARGMAX_LAYER_H
#define JAFFE_ARGMAX_LAYER_H

#include "layer.h"

namespace jaffe {
    // 计算每一个datum中最大值的index

    // 用于分类模型
    // 如果 m_out_val 为 ture， 则输出为一个 vector ， 内容为每一张图片的 paris (max_ind, max_val)
    // m_axis 指明了只选取其中一个 axis 进行求最大值

    template <typename Dtype>
    class JArgMaxLayer : public JLayer<Dtype>
    {
    public:
        explicit JArgMaxLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual  void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
                                 const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
                             const vector<JBlob<Dtype>*>& top);


        virtual inline const char* type() const { return "ArgMax"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }
    protected:

        virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
                             const vector<JBlob<Dtype>*>& top);

        bool m_out_max_val;    // m_out_max_val 如果为ture， 则输出为 paris (max_ind, max_val)
        size_t m_top_k;        // top_k  指定输出的排序后最大值的数量，默认为1
        bool m_has_axis;       //
        int m_axis;            // 在指定的 axis 上求去最大值
    };
}

#endif //JAFFE_ARGMAX_LAYER_H
