#include <algorithm>

#include "layers/slice_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JSliceLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const SliceParameter& slice_param = this->m_layer_param.slice_param();
//    CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
//    << "Either axis or slice_dim should be specified; not both.";
        m_slice_point.clear();
        std::copy(slice_param.slice_point().begin(), slice_param.slice_point().end(), std::back_inserter(m_slice_point));
    }

    template <typename Dtype>
    void JSliceLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const int num_axes = bottom[0]->GetNumAxes();
        const SliceParameter& slice_param = this->m_layer_param.slice_param();
        if (slice_param.has_slice_dim()) {
            m_slice_axis = static_cast<int>(slice_param.slice_dim());
// Don't allow negative indexing for slice_dim, a uint32 -- almost
// certainly unintended.
//CHECK_GE(m_slice_axis, 0) << "casting slice_dim from uint32 to int32 "
//<< "produced negative result; slice_dim must satisfy "
//<< "0 <= slice_dim < " << kMaxBlobAxes;
//CHECK_LT(m_slice_axis, num_axes) << "slice_dim out of range.";
        } else {
            m_slice_axis = bottom[0]->CanonicalAxisIndex(slice_param.axis());
        }
        vector<int> top_shape = bottom[0]->GetShape();
        const int bottom_slice_axis = bottom[0]->GetShape(m_slice_axis);
        m_num_slices = bottom[0]->GetCount(0, m_slice_axis);
        m_slice_size = bottom[0]->GetCount(m_slice_axis + 1);
        int count = 0;
        if (m_slice_point.size() != 0) {
//CHECK_EQ(m_slice_point.size(), top.size() - 1);
//CHECK_LE(top.size(), bottom_slice_axis);
            int prev = 0;
            vector<int> slices;
            for (int i = 0; i < m_slice_point.size(); ++i) {
//CHECK_GT(m_slice_point[i], prev);
                slices.push_back(m_slice_point[i] - prev);
                prev = m_slice_point[i];
            }
            slices.push_back(bottom_slice_axis - prev);
            for (int i = 0; i < top.size(); ++i) {
                top_shape[m_slice_axis] = slices[i];
                top[i]->Reshape(top_shape);
                count += top[i]->GetCount();
            }
        } else {
//CHECK_EQ(bottom_slice_axis % top.size(), 0)
//<< "Number of top blobs (" << top.size() << ") should evenly "
//<< "divide input slice axis (" << bottom_slice_axis << ")";
            top_shape[m_slice_axis] = bottom_slice_axis / top.size();
            for (int i = 0; i < top.size(); ++i) {
                top[i]->Reshape(top_shape);
                count += top[i]->GetCount();
            }
        }
//CHECK_EQ(count, bottom[0]->count());
        if (top.size() == 1) {
            top[0]->ShareData(*bottom[0]);
            top[0]->ShareDiff(*bottom[0]);
        }
    }

    template <typename Dtype>
    void JSliceLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        if (top.size() == 1) { return; }
        int offset_slice_axis = 0;
        const Dtype* bottom_data = bottom[0]->GetData();
        const int bottom_slice_axis = bottom[0]->GetShape(m_slice_axis);
        for (int i = 0; i < top.size(); ++i) {
            Dtype* top_data = top[i]->GetMutableData();
            const int top_slice_axis = top[i]->GetShape(m_slice_axis);
            for (int n = 0; n < m_num_slices; ++n) {
                const int top_offset = n * top_slice_axis * m_slice_size;
                const int bottom_offset = (n * bottom_slice_axis + offset_slice_axis) * m_slice_size;
                JaffeCopy(top_slice_axis * m_slice_size, bottom_data + bottom_offset, top_data + top_offset);
            }
            offset_slice_axis += top_slice_axis;
        }
    }

    INSTANTIATE_CLASS(JSliceLayer);
    REGISTER_LAYER_CLASS(Slice);

}
