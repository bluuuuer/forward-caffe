// Unfinished
#include "layers/im2col_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JIm2colLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        ConvolutionParameter conv_param = this->m_layer_param.convolution_param();
        m_force_nd_im2col = conv_param.force_nd_im2col();
        const int input_num_dims = bottom[0]->GetShape().size();
        m_channel_axis = bottom[0]->CanonicalAxisIndex(conv_param.axis());
        const int first_spatial_dim = m_channel_axis + 1;
        m_num_spatial_axes = input_num_dims - first_spatial_dim;
//    CHECK_GE(num_spatial_axes_, 1);
        vector<int> dim_blob_shape(1, m_num_spatial_axes);
        // Setup filter kernel dimensions (kernel_shape_).
        m_kernel_shape.Reshape(dim_blob_shape);
        int* kernel_shape_data = m_kernel_shape.GetMutableData();
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
//    CHECK_EQ(num_spatial_axes_, 2)
//    << "kernel_h & kernel_w can only be used for 2D convolution.";
//    CHECK_EQ(0, conv_param.kernel_size_size())
//    << "Either kernel_size or kernel_h/w should be specified; not both.";
            kernel_shape_data[0] = conv_param.kernel_h();
            kernel_shape_data[1] = conv_param.kernel_w();
        } else {
            const int num_kernel_dims = conv_param.kernel_size_size();
//CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
//<< "kernel_size must be specified once, or once per spatial dimension "
//<< "(kernel_size specified " << num_kernel_dims << " times; "
//<< num_spatial_axes_ << " spatial dims);";
            for (int i = 0; i < m_num_spatial_axes; ++i) {
                kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
            }
        }
        for (int i = 0; i < m_num_spatial_axes; ++i) {
//CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
        }
// Setup stride dimensions (stride_).
        m_stride.Reshape(dim_blob_shape);
        int* stride_data = m_stride.GetMutableData();
        if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
//CHECK_EQ(num_spatial_axes_, 2)
//<< "stride_h & stride_w can only be used for 2D convolution.";
//CHECK_EQ(0, conv_param.stride_size())
//<< "Either stride or stride_h/w should be specified; not both.";
            stride_data[0] = conv_param.stride_h();
            stride_data[1] = conv_param.stride_w();
        } else {
            const int num_stride_dims = conv_param.stride_size();
//CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
//num_stride_dims == num_spatial_axes_)
//<< "stride must be specified once, or once per spatial dimension "
//<< "(stride specified " << num_stride_dims << " times; "
//<< num_spatial_axes_ << " spatial dims);";
            const int kDefaultStride = 1;
            for (int i = 0; i < m_num_spatial_axes; ++i) {
                stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                                 conv_param.stride((num_stride_dims == 1) ? 0 : i);
//CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
            }
        }
// Setup pad dimensions (pad_).
        m_pad.Reshape(dim_blob_shape);
        int* pad_data = m_pad.GetMutableData();
        if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
//CHECK_EQ(num_spatial_axes_, 2)
//<< "pad_h & pad_w can only be used for 2D convolution.";
//CHECK_EQ(0, conv_param.pad_size())
//<< "Either pad or pad_h/w should be specified; not both.";
            pad_data[0] = conv_param.pad_h();
            pad_data[1] = conv_param.pad_w();
        } else {
            const int num_pad_dims = conv_param.pad_size();
//CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
//num_pad_dims == num_spatial_axes_)
//<< "pad must be specified once, or once per spatial dimension "
//<< "(pad specified " << num_pad_dims << " times; "
//<< num_spatial_axes_ << " spatial dims);";
            const int kDefaultPad = 0;
            for (int i = 0; i < m_num_spatial_axes; ++i) {
                pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                              conv_param.pad((num_pad_dims == 1) ? 0 : i);
            }
        }
// Setup dilation dimensions (dilation_).
        m_dilation.Reshape(dim_blob_shape);
        int* dilation_data = m_dilation.GetMutableData();
        const int num_dilation_dims = conv_param.dilation_size();
//CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
//num_dilation_dims == num_spatial_axes_)
//<< "dilation must be specified once, or once per spatial dimension "
//<< "(dilation specified " << num_dilation_dims << " times; "
//<< num_spatial_axes_ << " spatial dims).";
        const int kDefaultDilation = 1;
        for (int i = 0; i < m_num_spatial_axes; ++i) {
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                               conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
        }
    }

    template <typename Dtype>
    void JIm2colLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        vector<int> top_shape = bottom[0]->GetShape();
        const int* kernel_shape_data = m_kernel_shape.GetData();
        const int* stride_data = m_stride.GetData();
        const int* pad_data = m_pad.GetData();
        const int* dilation_data = m_dilation.GetData();
        for (int i = 0; i < m_num_spatial_axes; ++i) {
            top_shape[m_channel_axis] *= kernel_shape_data[i];
            const int input_dim = bottom[0]->GetShape(m_channel_axis + i + 1);
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                                   / stride_data[i] + 1;
            top_shape[m_channel_axis + i + 1] = output_dim;
        }
        top[0]->Reshape(top_shape);
        m_num = bottom[0]->GetCount(0, m_channel_axis);
        m_bottom_dim = bottom[0]->GetCount(m_channel_axis);
        m_top_dim = top[0]->GetCount(m_channel_axis);

        m_channels = bottom[0]->GetShape(m_channel_axis);
    }

    template <typename Dtype>
    void JIm2colLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        for (int n = 0; n < m_num; ++n)
        {
//DCHECK_EQ(bottom[0]->shape().size() - channel_axis_, num_spatial_axes_ + 1);
//DCHECK_EQ(top[0]->shape().size() - channel_axis_, num_spatial_axes_ + 1);
//DCHECK_EQ(kernel_shape_.count(), num_spatial_axes_);
//DCHECK_EQ(pad_.count(), num_spatial_axes_);
//DCHECK_EQ(stride_.count(), num_spatial_axes_);
//DCHECK_EQ(dilation_.count(), num_spatial_axes_);
            if (!m_force_nd_im2col && m_num_spatial_axes == 2)
            {
//im2col_cpu(bottom_data + n * m_bottom_dim, m_channels,
//           bottom[0]->GetShape(m_channel_axis + 1),
//bottom[0]->shape(m_channel_axis + 2),
//m_kernel_shape.GetData()[0], m_kernel_shape.GetData()[1],
//        m_pad.GetData()[0], m_pad.GetData()[1],
//        m_stride.GetData()[0], m_stride.GetData()[1],
//        m_dilation.GetData()[0], m_dilation.GetData()[1],
//        top_data + n * m_top_dim);
            } else {
//im2col_nd_cpu(bottom_data + n * m_bottom_dim, m_num_spatial_axes,
//        bottom[0]->GetShape().data() + m_channel_axis,
//top[0]->GetShape().data() + m_channel_axis,
//m_kernel_shape.GetData(), m_pad.GetData(), m_stride.GetData(),
//        m_dilation.GetData(), top_data + n * m_top_dim);
            }
        }
    }



    INSTANTIATE_CLASS(JIm2colLayer);
    REGISTER_LAYER_CLASS(Im2col);
}
