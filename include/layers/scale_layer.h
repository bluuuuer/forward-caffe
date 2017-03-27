#ifndef JAFFE_SCALE_LAYER_H
#define JAFFE_SCALE_LAYER_H

#include "layers/bias_layer.h"

namespace jaffe {

    // 和 bias_layer 很相似 ， 不过 scale_layer 计算的是 product， bias_layer 计算的是sum
/**
 * @brief Computes a product of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer.
 */
    template <typename Dtype>
    class JScaleLayer : public JLayer<Dtype>
    {
    public:
        explicit JScaleLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Scale"; }
        // Scale
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MaxBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        /**
         * In the below shape specifications, @f$ i @f$ denotes the value of the
         * `axis` field given by `this->layer_param_.scale_param().axis()`, after
         * canonicalization (i.e., conversion from negative to positive index,
         * if applicable).
         *
         * @param bottom input Blob vector (length 2)
         *   -# @f$ (d_0 \times ... \times
         *           d_i \times ... \times d_j \times ... \times d_n) @f$
         *      the first factor @f$ x @f$
         *   -# @f$ (d_i \times ... \times d_j) @f$
         *      the second factor @f$ y @f$
         * @param top output Blob vector (length 1)
         *   -# @f$ (d_0 \times ... \times
         *           d_i \times ... \times d_j \times ... \times d_n) @f$
         *      the product @f$ z = x y @f$ computed after "broadcasting" y.
         *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
         *      then computing the elementwise product.
         */
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);


        shared_ptr<JLayer<Dtype> > m_bias_layer;
        vector<JBlob<Dtype>*> m_bias_bottom_vec;
        vector<bool> m_bias_propagate_down;
        int m_bias_param_id;

        JBlob<Dtype> m_sum_multiplier;
        JBlob<Dtype> m_sum_result;
        JBlob<Dtype> m_temp;
        int m_axis;
        int m_outer_dim, m_scale_dim, m_inner_dim;
    };
}
#endif //JAFFE_SCALE_LAYER_H
