// unfinished
#include <algorithm>

#include "layers/spp_layer.h"
#include "layers/flatten_layer.h"
#include "layers/split_layer.h"
#include "layers/concat_layer.h"
#include "layers/pooling_layer.h"

namespace jaffe {
    using std::min;
    using std::max;

    template <typename Dtype>
    LayerParameter JSPPLayer<Dtype>::GetPoolingParam(const int pyramid_level,
                                                     const int bottom_h,
                                                     const int bottom_w,
                                                     const SPPParameter spp_param) {
        LayerParameter pooling_param;
        int num_bins = pow(2, pyramid_level);

        // find padding and kernel size so that the pooling is
        // performed across the entire image
        int kernel_h = ceil(bottom_h / static_cast<double>(num_bins));
        // remainder_h is the min number of pixels that need to be padded before
        // entire image height is pooled over with the chosen kernel dimension
        int remainder_h = kernel_h * num_bins - bottom_h;
        // pooling layer pads (2 * pad_h) pixels on the top and bottom of the
        // image.
        int pad_h = (remainder_h + 1) / 2;

        // similar logic for width
        int kernel_w = ceil(bottom_w / static_cast<double>(num_bins));
        int remainder_w = kernel_w * num_bins - bottom_w;
        int pad_w = (remainder_w + 1) / 2;

        pooling_param.mutable_pooling_param()->set_pad_h(pad_h);
        pooling_param.mutable_pooling_param()->set_pad_w(pad_w);
        pooling_param.mutable_pooling_param()->set_kernel_h(kernel_h);
        pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
        pooling_param.mutable_pooling_param()->set_stride_h(kernel_h);
        pooling_param.mutable_pooling_param()->set_stride_w(kernel_w);

        switch (spp_param.pool()) {
            case SPPParameter_PoolMethod_MAX:
                pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_MAX);
                break;
            case SPPParameter_PoolMethod_AVE:
                pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_AVE);
                break;
            case SPPParameter_PoolMethod_STOCHASTIC:
                pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
                break;
            default:
                ;
                //LOG(FATAL) << "Unknown pooling method.";
        }

        return pooling_param;
    }

    template <typename Dtype>
    void JSPPLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        SPPParameter spp_param = this->m_layer_param.spp_param();

        m_num = bottom[0]->GetNum();
        m_channels = bottom[0]->GetChannels();
        m_bottom_h = bottom[0]->GetHeight();
        m_bottom_w = bottom[0]->GetWidth();
        m_reshaped_first_time = false;
        //CHECK_GT(m_bottom_h, 0) << "Input dimensions cannot be zero.";
        //CHECK_GT(m_bottom_w, 0) << "Input dimensions cannot be zero.";

        m_pyramid_height = spp_param.pyramid_height();
        m_split_top_vec.clear();
        m_pooling_bottom_vecs.clear();
        m_pooling_layers.clear();
        m_pooling_top_vecs.clear();
        m_pooling_outputs.clear();
        m_flatten_layers.clear();
        m_flatten_top_vecs.clear();
        m_flatten_outputs.clear();
        m_concat_bottom_vec.clear();

        if (m_pyramid_height == 1) {
            // pooling layer setup
            LayerParameter pooling_param = GetPoolingParam(0, m_bottom_h, m_bottom_w, spp_param);
            m_pooling_layers.push_back(shared_ptr<JPoolingLayer<Dtype> > (new JPoolingLayer<Dtype>(pooling_param)));
            m_pooling_layers[0]->SetUp(bottom, top);
            return;
        }
// split layer output holders setup
        for (int i = 0; i < m_pyramid_height; i++) {
            m_split_top_vec.push_back(new JBlob<Dtype>());
        }

// split layer setup
        LayerParameter split_param;
        m_split_layer.reset(new JSplitLayer<Dtype>(split_param));
        m_split_layer->SetUp(bottom, m_split_top_vec);

        for (int i = 0; i < m_pyramid_height; i++) {
// pooling layer input holders setup
            m_pooling_bottom_vecs.push_back(new vector<JBlob<Dtype>*>);
            m_pooling_bottom_vecs[i]->push_back(m_split_top_vec[i]);

// pooling layer output holders setup
            m_pooling_outputs.push_back(new JBlob<Dtype>());
            m_pooling_top_vecs.push_back(new vector<JBlob<Dtype>*>);
            m_pooling_top_vecs[i]->push_back(m_pooling_outputs[i]);

// pooling layer setup
            LayerParameter pooling_param = GetPoolingParam(i, m_bottom_h, m_bottom_w, spp_param);

            m_pooling_layers.push_back(shared_ptr<JPoolingLayer<Dtype> > (new JPoolingLayer<Dtype>(pooling_param)));
            m_pooling_layers[i]->SetUp(*m_pooling_bottom_vecs[i], *m_pooling_top_vecs[i]);

// flatten layer output holders setup
            m_flatten_outputs.push_back(new JBlob<Dtype>());
            m_flatten_top_vecs.push_back(new vector<JBlob<Dtype>*>);
            m_flatten_top_vecs[i]->push_back(m_flatten_outputs[i]);

// flatten layer setup
            LayerParameter flatten_param;
            m_flatten_layers.push_back(new JFlattenLayer<Dtype>(flatten_param));
            m_flatten_layers[i]->SetUp(*m_pooling_top_vecs[i], *m_flatten_top_vecs[i]);

// concat layer input holders setup
            m_concat_bottom_vec.push_back(m_flatten_outputs[i]);
        }

// concat layer setup
        LayerParameter concat_param;
        m_concat_layer.reset(new JConcatLayer<Dtype>(concat_param));
        m_concat_layer->SetUp(m_concat_bottom_vec, top);
    }

    template <typename Dtype>
    void JSPPLayer<Dtype>::Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
//CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
//<< "corresponding to (num, channels, height, width)";
// Do nothing if bottom shape is unchanged since last Reshape
        if (m_num == bottom[0]->GetNum() && m_channels == bottom[0]->GetChannels() &&
            m_bottom_h == bottom[0]->GetHeight() && m_bottom_w == bottom[0]->GetWidth() &&
            m_reshaped_first_time)
        {
            return;
        }
        m_num = bottom[0]->GetNum();
        m_channels = bottom[0]->GetChannels();
        m_bottom_h = bottom[0]->GetHeight();
        m_bottom_w = bottom[0]->GetWidth();
        m_reshaped_first_time = true;
        SPPParameter spp_param = this->m_layer_param.spp_param();
        if (m_pyramid_height == 1) {
            LayerParameter pooling_param = GetPoolingParam(0, m_bottom_h, m_bottom_w, spp_param);
            m_pooling_layers[0].reset(new JPoolingLayer<Dtype>(pooling_param));
            m_pooling_layers[0]->SetUp(bottom, top);
            m_pooling_layers[0]->Reshape(bottom, top);
            return;
        }
        m_split_layer->Reshape(bottom, m_split_top_vec);
        for (int i = 0; i < m_pyramid_height; i++) {
            LayerParameter pooling_param = GetPoolingParam(i, m_bottom_h, m_bottom_w, spp_param);

            m_pooling_layers[i].reset(new JPoolingLayer<Dtype>(pooling_param));
            m_pooling_layers[i]->SetUp(*m_pooling_bottom_vecs[i], *m_pooling_top_vecs[i]);
            m_pooling_layers[i]->Reshape(*m_pooling_bottom_vecs[i], *m_pooling_top_vecs[i]);
            m_flatten_layers[i]->Reshape(*m_pooling_top_vecs[i], *m_flatten_top_vecs[i]);
        }
        m_concat_layer->Reshape(m_concat_bottom_vec, top);
    }

    template <typename Dtype>
    void JSPPLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        if (m_pyramid_height == 1) {
//            m_pooling_layers[0]->Forward(bottom, top);
            return;
        }
        //m_split_layer->Forward(bottom, m_split_top_vec);
        for (int i = 0; i < m_pyramid_height; i++) {
     //       m_pooling_layers[i]->Forward(*m_pooling_bottom_vecs[i], *m_pooling_top_vecs[i]);
   //         m_flatten_layers[i]->Forward(*m_pooling_top_vecs[i], *m_flatten_top_vecs[i]);
        }
      //  m_concat_layer->Forward(m_concat_bottom_vec, top);
    }


    INSTANTIATE_CLASS(JSPPLayer);
    REGISTER_LAYER_CLASS(SPP);
}
