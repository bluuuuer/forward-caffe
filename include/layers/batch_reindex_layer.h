

#ifndef JAFFE_BATCH_REINDEX_LAYER_H
#define JAFFE_BATCH_REINDEX_LAYER_H

#include <utility>

#include "layer.h"

namespace jaffe {

/**
 * @brief Index into the input blob along its first axis.
 *
 * This layer can be used to select, reorder, and even replicate examples in a
 * batch.  The second blob is cast to int and treated as an index into the
 * first axis of the first blob.
 */

    template <typename Dtype>
    class JBatchReindexLayer : public JLayer<Dtype>
    {
    public:
        explicit JBatchReindexLayer(const LayerParameter& param)
                : JLayer<Dtype>(param) {}
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "BatchReindex"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

    private:
        struct pair_sort_first {
            bool operator()(const std::pair<int, int> &left,
                            const std::pair<int, int> &right) {
                return left.first < right.first;
            }
        };
        void check_batch_reindex(int initial_num, int final_num,
                                 const Dtype* ridx_data);
    };

}
#endif //JAFFE_BATCH_REINDEX_LAYER_H
