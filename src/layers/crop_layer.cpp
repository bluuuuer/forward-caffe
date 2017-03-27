#include <algorithm>
#include <functional>
#include <map>
#include <set>

#include "layers/crop_layer.h"

namespace jaffe {
    template <typename Dtype>
    void JCropLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const CropParameter& param = this->m_layer_param.crop_param();
        int input_dim = bottom[0]->GetNumAxes();
        const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
        if (param.offset_size() > 1)
        {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
//    CHECK_EQ(start_axis + param.offset_size(), input_dim)
//    << "number of offset values specified must be equal to the number of "
//    << "dimensions following axis.";
        }
    }

    template <typename Dtype>
    void JCropLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom,
                                    const vector<JBlob<Dtype>*>& top)
    {
        const CropParameter& param = this->m_layer_param.crop_param();
        int input_dim = bottom[0]->GetNumAxes();
        const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

        // 初始化 offsets 为 0
        offsets = vector<int>(input_dim, 0);
        vector<int> new_shape(bottom[0]->GetShape());

        for (int i = 0; i < input_dim; ++i)
        {
            int crop_offset = 0;
            int new_size = bottom[0]->GetShape(i);
            if (i >= start_axis)
            {
                new_size = bottom[1]->GetShape(i);
                if (param.offset_size() == 1)
                {
                    crop_offset = param.offset(0);
                } else if (param.offset_size() > 1)
                {
                    crop_offset = param.offset(i - start_axis);
                }
// Check that the crop and offset are within the dimension's bounds.
//CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
//<< "the crop for dimension " << i << " is out-of-bounds with "
//<< "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
            }
            new_shape[i] = new_size;
            offsets[i] = crop_offset;
        }
        top[0]->Reshape(new_shape);
    }

    template <typename Dtype>
    void JCropLayer<Dtype>::crop_copy(const vector<JBlob<Dtype>*>& bottom,
                                      const vector<JBlob<Dtype>*>& top,
                                      const vector<int>& offsets,
                                      vector<int> indices,
                                      int cur_dim,
                                      const Dtype* src_data,
                                      Dtype* dest_data,
                                      bool is_forward)
    {
        if (cur_dim + 1 < top[0]->GetNumAxes())
        {
            // 还不在最后维度，递归调用 copy
            for (int i = 0; i < top[0]->GetShape(cur_dim); ++i) {
                indices[cur_dim] = i;
                crop_copy(bottom, top, offsets, indices, cur_dim+1, src_data, dest_data, is_forward);
            }
        } else { //cur_dim + 1 == top[0]->GetNumAxes()
            // 在最后维度，在内存中是连续存储的
            for (int i = 0; i < top[0]->GetShape(cur_dim); ++i) {
                // index vector reduced(red) & index vector with offsets(off)
                std::vector<int> ind_red(cur_dim, 0);
                std::vector<int> ind_off(cur_dim+1, 0);
                for (int j = 0; j < cur_dim; ++j) {
                    ind_red[j] = indices[j];
                    ind_off[j] = indices[j] + offsets[j];
                }
                ind_off[cur_dim] = offsets[cur_dim];
                // copy
                if (is_forward) {
                    JaffeCopy(top[0]->GetShape(cur_dim),
                              src_data + bottom[0]->GetOffset(ind_off),
                              dest_data + top[0]->GetOffset(ind_red));
                } else {// backwards

                    JaffeCopy(top[0]->GetShape(cur_dim),
                              src_data + top[0]->GetOffset(ind_red),
                              dest_data + bottom[0]->GetOffset(ind_off));
                }
            }
        }
    }

    template <typename Dtype>
    void JCropLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        std::vector<int> indices(top[0]->GetNumAxes(), 0);
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
    }

    INSTANTIATE_CLASS(JCropLayer);
    REGISTER_LAYER_CLASS(Crop);
}
