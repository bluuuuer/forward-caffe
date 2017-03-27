//
// Created by huangshize on 16-4-22.
//

#ifndef JAFFE_INSERT_SPLITS_H
#define JAFFE_INSERT_SPLITS_H

#include <string>

#include "common.h"

namespace jaffe {

    void InsertSplits(const NetParameter& param, NetParameter* param_split);

    void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
                             const int blob_idx, const int split_count, const float loss_weight,
                             LayerParameter* split_layer_param);

    string SplitLayerName(const string& layer_name, const string& blob_name,
                          const int blob_idx);

    string SplitBlobName(const string& layer_name, const string& blob_name,
                         const int blob_idx, const int split_idx);
} // namespace jaffe
#endif //JAFFE_INSERT_SPLITS_H
