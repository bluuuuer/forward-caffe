// huangshize 2016.03.28
// huangshize 2016.04.05 bug fixed
// huangshize 2016.04.09 bug fixed
// 注释了部分有关blob内元素的计算
#ifndef JAFFE_BLOB_H_
#define JAFFE_BLOB_H_

#include <iostream>

#include "common.h"
#include "rawdata.h"

namespace jaffe {

    template <typename Dtype>
    class JBlob
    {
    public:
        // 构造函数
        JBlob() : m_data(), m_count(0), m_capacity(0) {}
        explicit JBlob(const int num, const int channels, const int height, const int width)
                : m_capacity(0) {
            Reshape(num, channels, height, width);
        }
        explicit JBlob(const vector<int>& shape) : m_capacity(0) { Reshape(shape); }


        //  Reshape
        void Reshape(const int num, const int channels, const int height,
                     const int width)
        {
            vector<int> shape(4);
            shape[0] = num;
            shape[1] = channels;
            shape[2] = height;
            shape[3] = width;
            Reshape(shape);
        }
        void Reshape(const vector<int>& shape)
        {
            m_count = 1;
            m_shape.resize(shape.size());
            if (!m_shape_data || m_shape_data->GetSize() < shape.size() * sizeof(int)) {
                m_shape_data.reset(new JRawData(shape.size() * sizeof(int)));
            }

//            int* shape_data = static_cast<int*>(m_shape_data->GetMutableCpuData());
            for (int i = 0; i < shape.size(); i++) {
                m_count *= shape[i];
                m_shape[i] = shape[i];
 //               shape_data[i] = shape[i];  // 不懂
            }
            // 有待测试
            if (m_count > m_capacity) {
                m_capacity = m_count;
                m_data.reset(new JRawData(m_capacity * sizeof(Dtype)));
                //m_diff.reset(new JRawData(m_capacity * sizeof(Dtype)));
            }
        }

        void Reshape(const BlobShape& shape) {
            vector<int> shape_vec(shape.dim_size());
            for(int i = 0; i < shape.dim_size(); i++)
                shape_vec[i] = shape.dim(i);
            //=====three days!=====
            Reshape(shape_vec);
        }

        inline string GetShapeString() const {
            ostringstream stream;
            for (int i = 0; i < m_shape.size(); i++)
                stream << m_shape[i] << " ";
            stream << "(" << m_count << ")";
            return stream.str();
        }

        void ReshapeLike(const JBlob& other);

        //
        inline const vector<int>& GetShape() const { return m_shape; }
		inline void SetNum(const int new_num) {
			vector<int> new_shape = m_shape;
			new_shape[0] = new_num;
			Reshape(new_shape);
		}
        // 输出第index个维度的维数，支持index小于0，反序查找
        inline int GetShape(int index) const
        {
            return m_shape[CanonicalAxisIndex(index)];
        }
        inline int CanonicalAxisIndex(int axis_index) const
        {
            if (axis_index < 0)
            {
                return axis_index + GetNumAxes();  //hsz0405 bug fixed
            }
            return axis_index;
        }
        // 获取数据指定维度的维数
        inline int GetNum() const { return LegacyShape(0); }
        inline int GetChannels() const { return LegacyShape(1); }
        inline int GetHeight() const { return LegacyShape(2); }
        inline int GetWidth() const { return LegacyShape(3); }
        inline int LegacyShape(int index) const {
            if (index >= GetNumAxes() || index < -GetNumAxes())
                return 1;
            return GetShape(index);
        }

        inline int GetNumAxes() const { return m_shape.size(); }
        inline int GetCount() const { return m_count; }
        inline int GetCount(int start_axis, int end_axis) const
        {
            // 需要判断index范围
            int count = 1;
            for (int i = start_axis; i < end_axis; i++) {
                count *= GetShape(i);
            }

            return count;
        }
        inline int GetCount(int start_axis) const
        {
            // 需要判断index范围
            return GetCount(start_axis, GetNumAxes());
        }

        // 计算 offset 偏移量
        // xw CalOffset => GetOffset @ 04.21
        inline int GetOffset(const int n, const int c = 0, const int h = 0,
                             const int w = 0) const
        {
            // 需要判断 n c h w 的大小有没有超出范围
            return ((n * GetChannels() + c) * GetHeight() + h) * GetWidth() + w;  // hsz0405 bug fixed
        }

        inline int GetOffset(const vector<int>& indices) const
        {
            int offset = 0;
            for (int i = 0; i < GetNumAxes(); i++)
            {
                offset *= GetShape(i);
                if (indices.size() > i)
                {
                    offset += indices[i];  // hsz0405 bug fixed
                }
            }
            return offset;
        }

        void CopyFrom(const JBlob<Dtype>& source, bool copy_diff = false, bool reshape = false);

        // 通过offset偏移来求具体数据
        inline Dtype DataAt(const int n, const int c, const int h, const int w) const
        {
            return GetData()[GetOffset(n, c, h, w)]; // hsz0405 bug fixed
        }

        inline Dtype DiffAt(const int n, const int c, const int h, const int w) const
        {
            return GetDiff()[GetOffset(n, c, h, w)]; // hsz0405 bug fixed
        }

        inline Dtype DataAt(const vector<int>& index) const
        {
            return GetData()[GetOffset(index)]; // hsz0405 bug fixed
        }

        inline Dtype DiffAt(const vector<int>& index) const
        {
            return GetDiff()[GetOffset(index)]; // hsz0405 bug fixed
        }

        inline const shared_ptr<JRawData>& GetBlobData() const
        {
            return m_data;
        } // 原 data()
        /*
		inline const shared_ptr<JRawData>& GetBlobDiff() const
        {
            return m_diff;
        }
		*/

        // xw Data => GetData @ 04.21
        const Dtype* GetData() const;	// 原blob.hpp中的cpu_data() / gpu_data()
		const Dtype* GetGpuData() const;
        const Dtype* GetDiff() const;	// 原blob.hpp中的cpu_diff() / gpu_diff()
		const Dtype* GetGpuDiff() const;
        Dtype* GetMutableData();
		Dtype* GetMutableGpuData();
        Dtype* GetMutableDiff();
		Dtype* GetMutableGpuDiff();
		const int* GetGpuShape() const;

        void SetData(Dtype* data);

        void Update();
        void FromProto(const BlobProto& proto, bool reshape = true);
        void ToProto(BlobProto* proto, bool write_diff = false) const;

        // 分别计算data和diff的绝对值之和以及平方之和
        Dtype AsumData() const;
        Dtype AsumDiff() const;
        Dtype SumsqData() const;
        Dtype SumsqDiff() const;
//
//		// 涉及cblas函数运算
//		void ScaleData(Dtype scale_factor);
//		void ScaleDiff(Dtype scale_factor);
//
//		//
        void ShareData(const JBlob& other);
        void ShareDiff(const JBlob& other);

        bool ShapeEquals(const BlobProto& other);

		inline void GetTag() {
			if (m_data.get()) {
				m_data->GetTag();
			}
		}

    protected:
        shared_ptr<JRawData> m_data;
 //       shared_ptr<JRawData> m_diff;
        shared_ptr<JRawData> m_shape_data;
        vector<int> m_shape;
        int m_count;
        int m_capacity;  // 容量  暂时保留

    };

}

#endif
