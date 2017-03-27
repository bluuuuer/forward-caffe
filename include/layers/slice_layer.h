
#ifndef JAFFE_SLICE_LAYER_H
#define JAFFE_SLICE_LAYER_H

#include "layer.h"

namespace jaffe {
/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 */
    template <typename Dtype>
    class JSliceLayer : public JLayer <Dtype>
    {
    public:
        explicit JSliceLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Slice"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        int m_count;
        int m_num_slices;
        int m_slice_size;
        int m_slice_axis;
        vector<int> m_slice_point;
    };
}
#endif //JAFFE_SLICE_LAYER_H
