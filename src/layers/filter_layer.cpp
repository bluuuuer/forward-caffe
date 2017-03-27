#include <vector>

#include "layers/filter_layer.h"

namespace jaffe {
    template <typename Dtype>
    void JFilterLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        m_first_reshape = true;
    }

    template <typename Dtype>
    void JFilterLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
// bottom[0...k-1] are the blobs to filter
// bottom[last] is the "selector_blob"
        int selector_index = bottom.size() - 1;
        for (int i = 1; i < bottom[selector_index]->GetNumAxes(); ++i) {
//CHECK_EQ(bottom[selector_index]->shape(i), 1)
//<< "Selector blob dimensions must be singletons (1), except the first";
        }
        for (int i = 0; i < bottom.size() - 1; ++i) {
//CHECK_EQ(bottom[selector_index]->shape(0), bottom[i]->shape(0)) <<
//"Each bottom should have the same 0th dimension as the selector blob";
        }

        const Dtype* bottom_data_selector = bottom[selector_index]->GetData();
        m_indices_to_forward.clear();

// look for non-zero elements in bottom[0]. Items of each bottom that
// have the same index as the items in bottom[0] with value == non-zero
// will be forwarded
        for (int item_id = 0; item_id < bottom[selector_index]->GetShape(0); ++item_id) {
// we don't need an offset because item size == 1
            const Dtype* tmp_data_selector = bottom_data_selector + item_id;
            if (*tmp_data_selector) {
                m_indices_to_forward.push_back(item_id); // 仅仅保留非0元素对应的id
            }
        }
// only filtered items will be forwarded
        int new_tops_num = m_indices_to_forward.size();
// init
        if (m_first_reshape) {
            new_tops_num = bottom[0]->GetShape(0);
            m_first_reshape = false;
        }
        for (int t = 0; t < top.size(); ++t) {
            int num_axes = bottom[t]->GetNumAxes();
            vector<int> shape_top(num_axes);
            shape_top[0] = new_tops_num;
            for (int ts = 1; ts < num_axes; ++ts)
                shape_top[ts] = bottom[t]->GetShape(ts);
            top[t]->Reshape(shape_top);
        }
    }

    template <typename Dtype>
    void JFilterLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        int new_tops_num = m_indices_to_forward.size();
// forward all filtered items for all bottoms but the Selector (bottom[last])
        for (int t = 0; t < top.size(); ++t) {
            const Dtype* bottom_data = bottom[t]->GetData();
            Dtype* top_data = top[t]->GetMutableData();
            int dim = bottom[t]->GetCount() / bottom[t]->GetShape(0);
            for (int n = 0; n < new_tops_num; ++n) {
                int data_offset_top = n * dim;
                int data_offset_bottom = m_indices_to_forward[n] * bottom[t]->GetCount(1);
                JaffeCopy(dim, bottom_data + data_offset_bottom, top_data + data_offset_top);
            }
        }
    }
    INSTANTIATE_CLASS(JFilterLayer);
    REGISTER_LAYER_CLASS(Filter);
}
