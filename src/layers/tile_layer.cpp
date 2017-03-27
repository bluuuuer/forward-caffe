#include "layers/tile_layer.h"

namespace jaffe {

    template <typename Dtype>
    void JTileLayer<Dtype>::Reshape(
            const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const TileParameter& tile_param = this->m_layer_param.tile_param();
        m_axis = bottom[0]->CanonicalAxisIndex(tile_param.axis());
        //CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
        m_tiles = tile_param.tiles();
        //CHECK_GT(m_tiles, 0) << "Number of tiles must be positive.";
        vector<int> top_shape = bottom[0]->GetShape();
        // 重复复制 tiles 份数据
        top_shape[m_axis] = bottom[0]->GetShape(m_axis) * m_tiles;
        top[0]->Reshape(top_shape);
        m_outer_dim = bottom[0]->GetCount(0, m_axis);
        m_inner_dim = bottom[0]->GetCount(m_axis);
    }

    template <typename Dtype>
    void JTileLayer<Dtype>::Forward(const vector<JBlob<Dtype>*>& bottom, const vector<JBlob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->GetData();
        Dtype* top_data = top[0]->GetMutableData();
        for (int i = 0; i < m_outer_dim; ++i) {
            for (int t = 0; t < m_tiles; ++t) {
                JaffeCopy(m_inner_dim, bottom_data, top_data);
                top_data += m_inner_dim;
            }
            bottom_data += m_inner_dim;
        }
    }

    INSTANTIATE_CLASS(JTileLayer);
    REGISTER_LAYER_CLASS(Tile);
}
