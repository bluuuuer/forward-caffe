//
// Created by huangshize on 16-5-18.
//

#ifndef JAFFE_CROP_LAYER_H
#define JAFFE_CROP_LAYER_H

#include <utility>

#include "layer.h"

namespace jaffe {

    /**
 * @brief Takes a Blob and crop it, to the shape specified by the second input
 *  Blob, across all dimensions after the specified axis.
 *
 */
    template <typename Dtype>
    class JCropLayer : public JLayer<Dtype>
    {
    public:
        explicit JCropLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Crop"; }
        // bottom[0] 提供数据 bottom[1] 提供尺寸
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        vector<int> offsets;

    private:
        // Recursive copy function.
        void crop_copy(const vector<JBlob<Dtype>*>& bottom,
                       const vector<JBlob<Dtype>*>& top,
                       const vector<int>& offsets,
                       vector<int> indices,
                       int cur_dim,
                       const Dtype* src_data,
                       Dtype* dest_data,
                       bool is_forward);
    };
}
#endif //JAFFE_CROP_LAYER_H
