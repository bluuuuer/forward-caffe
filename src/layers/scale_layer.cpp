// Unfinished
// math
// Forward
#include <algorithm>

#include "filler.h"
#include "layer_factory.h"
#include "layers/scale_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JScaleLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const ScaleParameter& param = this->m_layer_param.scale_param();
        if (bottom.size() == 1 && this->m_blobs.size() > 0) {
//    LOG(INFO) << "Skipping parameter initialization";
        } else if (bottom.size() == 1) {
// scale is a learned parameter; initialize it
            m_axis = bottom[0]->CanonicalAxisIndex(param.axis());
            const int num_axes = param.num_axes();
//CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
//<< "or -1 to extend to the end of bottom[0]";
            if (num_axes >= 0) {
//CHECK_GE(bottom[0]->num_axes(), m_axis + num_axes)
//<< "scale blob's shape extends past bottom[0]'s shape when applied "
//<< "starting with bottom[0] axis = " << m_axis;
            }
            this->m_blobs.resize(1);
            const vector<int>::const_iterator& shape_start = bottom[0]->GetShape().begin() + m_axis;
            const vector<int>::const_iterator& shape_end = (num_axes == -1) ? bottom[0]->GetShape().end() : (shape_start + num_axes);
            vector<int> scale_shape(shape_start, shape_end);
            this->m_blobs[0].reset(new JBlob<Dtype>(scale_shape));
            FillerParameter filler_param(param.filler());
            if (!param.has_filler()) {
// Default to unit (1) filler for identity operation.
                filler_param.set_type("constant");
                filler_param.set_value(1);
            }
            shared_ptr<JFiller<Dtype> > filler(GetFiller<Dtype>());
            filler->Fill(this->m_blobs[0].get());
        }
        if (param.bias_term()) {
            LayerParameter layer_param(this->m_layer_param);
            layer_param.set_type("Bias");
            BiasParameter* bias_param = layer_param.mutable_bias_param();
            bias_param->set_axis(param.axis());
            if (bottom.size() > 1) {
                bias_param->set_num_axes(bottom[1]->GetNumAxes());
            } else {
                bias_param->set_num_axes(param.num_axes());
            }
            bias_param->mutable_filler()->CopyFrom(param.bias_filler());
            m_bias_layer = JLayerRegistry<Dtype>::CreateLayer(layer_param);
            m_bias_bottom_vec.resize(1);
            m_bias_bottom_vec[0] = bottom[0];
            m_bias_layer->SetUp(m_bias_bottom_vec, top);
            m_bias_param_id = this->m_blobs.size();
            this->m_blobs.resize(m_bias_param_id + 1);
            this->m_blobs[m_bias_param_id] = m_bias_layer->GetBlobs()[0];
            m_bias_propagate_down.resize(1, false);
        }
        this->m_param_propagate_down.resize(this->m_blobs.size(), false);
    }

    template <typename Dtype>
    void JScaleLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const ScaleParameter& param = this->m_layer_param.scale_param();
        JBlob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->m_blobs[0].get();
// Always set axis_ == 0 in special case where scale is a scalar
// (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
// actual setting can be safely ignored; and computation is most efficient
// with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
// bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
// performant.)
        m_axis = (scale->GetNumAxes() == 0) ?
                 0 : bottom[0]->CanonicalAxisIndex(param.axis());
//CHECK_GE(bottom[0]->num_axes(), m_axis + scale->num_axes())
//<< "scale blob's shape extends past bottom[0]'s shape when applied "
//<< "starting with bottom[0] axis = " << m_axis;
        for (int i = 0; i < scale->GetNumAxes(); ++i) {
//CHECK_EQ(bottom[0]->shape(m_axis + i), scale->shape(i))
//<< "dimension mismatch between bottom[0]->shape(" << m_axis + i
//<< ") and scale->shape(" << i << ")";
        }
        m_outer_dim = bottom[0]->GetCount(0, m_axis);
        m_scale_dim = scale->GetCount();
        m_inner_dim = bottom[0]->GetCount(m_axis + scale->GetNumAxes());
        if (bottom[0] == top[0]) {  // in-place computation
            m_temp.ReshapeLike(*bottom[0]);
        } else {
            top[0]->ReshapeLike(*bottom[0]);
        }
        m_sum_result.Reshape(vector<int>(1, m_outer_dim * m_scale_dim));
        const int sum_mult_size = std::max(m_outer_dim, m_inner_dim);
        m_sum_multiplier.Reshape(vector<int>(1, sum_mult_size));
        if (m_sum_multiplier.GetData()[sum_mult_size - 1] != Dtype(1)) {
            JaffeSet(sum_mult_size, Dtype(1), m_sum_multiplier.GetMutableData());
        }
        if (m_bias_layer) {
            m_bias_bottom_vec[0] = top[0];
            m_bias_layer->Reshape(m_bias_bottom_vec, top);
        }
    }

    template <typename Dtype>
    void JScaleLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        if (bottom[0] == top[0]) {
// In-place computation; need to store bottom data before overwriting it.
// Note that this is only necessary for Backward; we could skip this if not
// doing Backward, but Caffe currently provides no way of knowing whether
// we'll need to do Backward at the time of the Forward call.
            JaffeCopy(bottom[0]->GetCount(), bottom[0]->GetData(), m_temp.GetMutableData());
        }
        const Dtype* scale_data = ((bottom.size() > 1) ? bottom[1] : this->m_blobs[0].get())->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        for (int n = 0; n < m_outer_dim; ++n) {
            for (int d = 0; d < m_scale_dim; ++d) {
                const Dtype factor = scale_data[d];
//caffe_cpu_scale(m_inner_dim, factor, bottom_data, top_data);
                bottom_data += m_inner_dim;
                top_data += m_inner_dim;
            }
        }
        if (m_bias_layer) {
            //m_bias_layer->Forward(m_bias_bottom_vec, top);
        }
    }
    INSTANTIATE_CLASS(JScaleLayer);
    REGISTER_LAYER_CLASS(Scale);

}
