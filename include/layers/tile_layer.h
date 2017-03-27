
#ifndef JAFFE_TILE_LAYER_H
#define JAFFE_TILE_LAYER_H

#include "layer.h"

namespace jaffe {
    // 在指定的维度重复复制blob数据
    template <typename Dtype>
    class JTileLayer : public JLayer<Dtype>
    {
    public:
        explicit JTileLayer(const LayerParameter& param) : JLayer<Dtype>(param) {}
        virtual void Reshape(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);

        virtual inline const char* type() const { return "Tile"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top);


        unsigned int m_axis, m_tiles, m_outer_dim, m_inner_dim;
    };
}
#endif //JAFFE_TILE_LAYER_H
