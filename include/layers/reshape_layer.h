#ifndef JAFFE_RESHAPE_LAYER_H
#define JAFFE_RESHAPE_LAYER_H

#include "layer.h"

namespace jaffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
    template <typename Dtype>
    class JReshapeLayer : public JLayer<Dtype> {
    public:
        explicit JReshapeLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Reshape"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top) {}

        /// @brief vector of axes indices whose dimensions we'll copy from the bottom
        vector<int> m_copy_axes;
        /// @brief the index of the axis whose dimension we infer, or -1 if none
        int m_inferred_axis;
        /// @brief the product of the "constant" output dimensions
        int m_constant_count;
    };

}
#endif //JAFFE_RESHAPE_LAYER_H
