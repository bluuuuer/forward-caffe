//
// Created by huangshize on 16-5-13.
//

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "layers/argmax_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JArgMaxLayer<Dtype>::LayerSetUp(const vector<JBlob<Dtype> *> &bottom,
                                         const vector<JBlob<Dtype> *> &top)
    {
        const ArgMaxParameter& argmax_param = this->m_layer_param.argmax_param();
        m_out_max_val = argmax_param.out_max_val();
        m_top_k = argmax_param.top_k();
        m_has_axis = argmax_param.has_axis();
        if (m_has_axis)
        {
            m_axis = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
        }
    }

    template <typename Dtype>
    void JArgMaxLayer<Dtype>::Reshape(const vector<JBlob<Dtype> *> &bottom,
                                      const vector<JBlob<Dtype> *> &top)
    {
        int num_top_axes = bottom[0]->GetNumAxes();
        if (num_top_axes < 3)
        {
            num_top_axes = 3;
        }
        std::vector<int> shape(num_top_axes, 1); // 1 1 1 ...
        if (m_has_axis)
        {
            shape = bottom[0]->GetShape();
            shape[m_axis] = m_top_k;
        }
        else
        {
            shape[0] = bottom[0]->GetShape(0);  // num
            shape[2] = m_top_k;
            if (m_out_max_val)  // 输出为 paris (max_ind, max_val)
            {
                shape[1] = 2;
            }
        }

        top[0]->Reshape(shape);
    }

    template <typename Dtype>
    void JArgMaxLayer<Dtype>::Forward(const vector<JBlob<Dtype> *> &bottom,
                                      const vector<JBlob<Dtype> *> &top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        int dim;
        int axis_dist;
        if (m_has_axis)
        {
            dim = bottom[0]->GetShape(m_axis);
            axis_dist = bottom[0]->GetCount(m_axis) / dim;
        }
        else
        {
            dim = bottom[0]->GetCount(1);
            axis_dist = 1;
        }
        int num = bottom[0]->GetCount() / dim;
        std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
        for (int i = 0; i < num; ++i)
        {

            for (int j = 0; j < dim; ++j)
            {
                bottom_data_vector[j] =
                        std::make_pair(bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
            }
            std::partial_sort(bottom_data_vector.begin(), bottom_data_vector.begin() + m_top_k,
                              bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

            for (int j = 0; j < m_top_k; ++j)
            {
                if(m_out_max_val)
                {
                    if (m_has_axis)
                    {
                        top_data[(i / axis_dist * m_top_k + j) * axis_dist + i % axis_dist]
                                = bottom_data_vector[j].first;
                    }
                    else
                    {
                        top_data[2 * i * m_top_k + j] = bottom_data_vector[j].second;
                        top_data[2 * i * m_top_k + m_top_k + j] = bottom_data_vector[j].first;
                    }

                }
                else
                {
                    top_data[(i / axis_dist * m_top_k + j) * axis_dist + i % axis_dist]
                            = bottom_data_vector[j].second;
                }

            }
        }

    }


    INSTANTIATE_CLASS(JArgMaxLayer);
    REGISTER_LAYER_CLASS(ArgMax);

}
