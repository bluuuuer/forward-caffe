#include "layers/batch_reindex_layer.h"

namespace jaffe {
    template<typename Dtype>
    void JBatchReindexLayer<Dtype>::Reshape(const vector<JBlob<Dtype> *> &bottom, const vector<JBlob<Dtype> *> &top)
    {
        vector<int> newshape;
        newshape.push_back(bottom[1]->GetShape(0));
        for (int i = 1; i < bottom[0]->GetShape().size(); ++i) {
            newshape.push_back(bottom[0]->GetShape()[i]);
        }
        top[0]->Reshape(newshape);
    }

    template<typename Dtype>
    void JBatchReindexLayer<Dtype>::check_batch_reindex(int initial_num, int final_num, const Dtype *ridx_data)
    {
        for (int i = 0; i < final_num; ++i) {
//            CHECK_GE(ridx_data[i], 0)
//            << "Index specified for reindex layer was negative.";
//            CHECK_LT(ridx_data[i], initial_num)
//            << "Index specified for reindex layer was greater than batch size.";
        }
    }

    template<typename Dtype>
    void JBatchReindexLayer<Dtype>::Forward(const vector<JBlob<Dtype> *> &bottom, const vector<JBlob<Dtype> *> &top)
    {
        check_batch_reindex(bottom[0]->GetShape(0), bottom[1]->GetCount(), bottom[1]->GetData());
        if (top[0]->GetCount() == 0) {
            return;
        }
        int inner_dim = bottom[0]->GetCount() / bottom[0]->GetShape(0);
        const Dtype *in = bottom[0]->GetData();
        const Dtype *permut = bottom[1]->GetData();
        Dtype *out = top[0]->GetMutableData();
        for (int index = 0; index < top[0]->GetCount(); ++index)
        {
            int n = index / (inner_dim);
            int in_n = static_cast<int>(permut[n]);
            out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
        }
    }
    INSTANTIATE_CLASS(JBatchReindexLayer);
    REGISTER_LAYER_CLASS(BatchReindex);

}



