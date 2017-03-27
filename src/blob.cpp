// huangshize 2016.03.28
// hsz0405 fix
// huangshize 2016.04.09 bug fixed
// 注释了部分有关blob内元素的计算
#include "blob.h"

namespace jaffe{

    template <typename Dtype>
    void JBlob<Dtype>::ReshapeLike(const JBlob<Dtype>& other) {
        Reshape(other.GetShape());
    }

    template <typename Dtype>
    void JBlob<Dtype>::CopyFrom(const JBlob& source, bool copy_diff, bool reshape) {
        if (source.GetCount() != m_count || source.GetShape() != m_shape)
        if (reshape)
            ReshapeLike(source);
     //   if (copy_diff)
//            JaffeCopy(m_count, source.GetDiff(),
 //                     static_cast<Dtype*>(m_diff->GetMutableCpuData()));
     //   else
	 	if (!copy_diff)
            JaffeCopy(m_count, source.GetData(),
                      static_cast<Dtype*>(m_data->GetMutableCpuData()));
    }

	template <typename Dtype>
	const int* JBlob<Dtype>::GetGpuShape() const {
		return (const int*)m_shape_data->GetGpuData();
	}

    template <typename Dtype>
    const Dtype* JBlob<Dtype>::GetData() const
    {
        return (const Dtype*)m_data->GetCpuData();
    }

	template <typename Dtype>
	const Dtype* JBlob<Dtype>::GetGpuData() const {
		return (const Dtype*)m_data->GetGpuData();
	}
    template <typename Dtype>
    const Dtype* JBlob<Dtype>::GetDiff() const
    {
     //   return (const Dtype*)m_diff->GetCpuData();
    }

	template <typename Dtype>
	const Dtype* JBlob<Dtype>::GetGpuDiff() const {
	//	return (const Dtype*)m_diff->GetGpuData();
	}

    template <typename Dtype>
    Dtype* JBlob<Dtype>::GetMutableData()
    {
        return static_cast<Dtype*>(m_data->GetMutableCpuData());
    }

	template <typename Dtype>
	Dtype* JBlob<Dtype>::GetMutableGpuData() {
		return static_cast<Dtype*>(m_data->GetMutableGpuData());
	}

    template <typename Dtype>
    Dtype* JBlob<Dtype>::GetMutableDiff()
    {
     //   return static_cast<Dtype*>(m_diff->GetMutableCpuData());
    }

    template <typename Dtype>
    Dtype* JBlob<Dtype>::GetMutableGpuDiff() {
 //       return static_cast<Dtype*>(m_diff->GetMutableGpuData());
	}

    template <typename Dtype>
    void JBlob<Dtype>::SetData(Dtype* data)
    {
        m_data->SetCpuData(data);
    }

    template <typename Dtype>
    void JBlob<Dtype>::Update() {
        if (m_data.get()->GetCpuData());
        //JaffeAxpy<Dtype>(m_count, Dtype(-1),
        //static_cast<const Dtype*>(m_diff->GetData()),
        //static_cast<Dtype*>(m_data->GetMutableData()));
    }

    template <typename Dtype>
    void JBlob<Dtype>::ShareData(const JBlob& other){
        // 需要判断两个JBlob的数据大小是否一致
        m_data = other.GetBlobData();
    }

    template <typename Dtype>
    void JBlob<Dtype>::ShareDiff(const JBlob& other){
 //       m_diff = other.GetBlobDiff();
    }

    template <typename Dtype>
    bool JBlob<Dtype>::ShapeEquals(const BlobProto& other) {
        if (other.has_num() || other.has_channels() || other.has_height() ||
            other.has_width())
            return m_shape.size() <= 4 &&
                   LegacyShape(-4) == other.num() &&
                   LegacyShape(-3) == other.channels() &&
                   LegacyShape(-2) == other.height() &&
                   LegacyShape(-1) == other.width();
        vector<int> other_shape(other.shape().dim_size());
        for (int i = 0; i < other.shape().dim_size(); i++)
            other_shape[i] = other.shape().dim(i);
        return m_shape == other_shape;
    }

    template <typename Dtype>
    void JBlob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
        //std::cout << "blob.cpp->FromProto()" << std::endl;
        if (reshape) {
            vector<int> shape;
            if (proto.has_num() || proto.has_channels() || proto.has_height() ||
                proto.has_width()) {
                shape.resize(4);
                shape[0] = proto.num();
                shape[1] = proto.channels();
                shape[2] = proto.height();
                shape[3] = proto.width();
            }
            else {
                shape.resize(proto.shape().dim_size());
                for (int i = 0; i < proto.shape().dim_size(); i++)
                    shape[i] = proto.shape().dim(i);
            }
			//cout << "FormProto() Reshape()" << endl;
            Reshape(shape);
        }
        else
            ShapeEquals(proto);

        Dtype* data_vec = GetMutableData();
        if (proto.double_data_size() > 0) {
            for (int i = 0; i < m_count; i ++) {
                data_vec[i] = proto.double_data(i);
            }
        }
        else {
            for (int i = 0; i < m_count; i++) {
                data_vec[i] = proto.data(i);
            }
        }

        Dtype* diff_vec = GetMutableDiff();
        if (proto.double_diff_size() > 0) {
            for (int i = 0; i < m_count; i++)
                diff_vec[i] = proto.double_diff(i);
        }
        else if (proto.diff_size() > 0) {
            for (int i = 0; i < m_count; i++)
                diff_vec[i] = proto.diff(i);
        }
    }

    template <>
    void JBlob<double>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for (int i = 0; i < m_shape.size(); i++)
            proto->mutable_shape()->add_dim(m_shape[i]);
        proto->clear_double_data();
        proto->clear_double_diff();
        const double* data_vec = GetData();
        for (int i = 0; i < m_count; i++)
            proto->add_double_data(data_vec[i]);
        if (write_diff) {
            const double* diff_vec = GetDiff();
            for (int i = 0; i < m_count; i++)
                proto->add_double_diff(diff_vec[i]);
        }
    }

    template <>
    void JBlob<float>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for (int i = 0; i < m_shape.size(); i++)
            proto->mutable_shape()->add_dim(m_shape[i]);
        proto->clear_data();
        proto->clear_diff();
        const float* data_vec = GetData();
        for (int i = 0; i < m_count; i++)
            proto->add_data(data_vec[i]);
        if (write_diff) {
            const float* diff_vec = GetDiff();
            for (int i = 0; i < m_count; i++)
                proto->add_diff(diff_vec[i]);
        }
    }

    template class JBlob<int>;
    template class JBlob<unsigned int>;
    template class JBlob<float>;
    template class JBlob<double>;

} // namespace jaffe
