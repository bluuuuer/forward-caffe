#ifndef JAFFE_IM2COL_LAYER_H
#define JAFFE_IM2COL_LAYER_H

#include "layer.h"

namespace jaffe {

    /**
 * @brief A helper for image operations that rearranges image regions into
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 */
    template <typename Dtype>
    class JIm2colLayer : public JLayer<Dtype> {
    public:
        explicit JIm2colLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Im2col"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        /// @brief The spatial dimensions of a filter kernel.
        JBlob<int> m_kernel_shape;
        /// @brief The spatial dimensions of the stride.
        JBlob<int> m_stride;
        /// @brief The spatial dimensions of the padding.
        JBlob<int> m_pad;
        /// @brief The spatial dimensions of the dilation.
        JBlob<int> m_dilation;

        int m_num_spatial_axes;
        int m_bottom_dim;
        int m_top_dim;

        int m_channel_axis;
        int m_num;
        int m_channels;

        bool m_force_nd_im2col;
    };
}
#endif //JAFFE_IM2COL_LAYER_H
