// huangshize 2016.04.06
// 增加了GetName()
// huangshize 2016.04.07
// GetBottomName(); GetBottomNum(); GetTopName();GetTopNum()
// huangshize 2016.04.08
// 把m_param修改为普通成员
#ifndef LAYER_H_H
#define LAYER_H_H

#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <sstream>
#include <memory>
#include <cfloat>
#include <assert.h>

#include "blob.h"
#include "util/im2col.h"
#include "layer_factory.h"
#include "filler.h"

using namespace std;

namespace jaffe {

	template <typename Dtype>
	class JLayer{
	public:
		explicit JLayer(const LayerParameter& param)
			: m_layer_param(param), m_is_shared(false){
			if (m_layer_param.blobs_size() > 0) {
				m_blobs.resize(m_layer_param.blobs_size());
				for (int i = 0; i < m_layer_param.blobs_size(); i++) {
					m_blobs[i].reset(new JBlob<Dtype>());
					m_blobs[i]->FromProto(m_layer_param.blobs(i));
				}
			}
		}
		virtual ~JLayer() {}

		void SetUp(const vector<JBlob<Dtype>*>& bottom,
						const vector<JBlob<Dtype>*>& top) {
			//InitMutex();
			CheckBlobCounts(bottom, top);
			LayerSetUp(bottom, top);
			Reshape(bottom, top);
			SetLossWeights(top);
		}

		virtual void CheckBlobCounts(const vector<JBlob<Dtype>*>& bottom,
						const vector<JBlob<Dtype>*>& top) {
			// unfinished
		}

		inline void SetLossWeights(const vector<JBlob<Dtype>*>& top) {
			//cout << "JLayer::SetLossWeights()" << endl;
			if (m_layer_param.loss_weight_size())
				for (int i = 0; i < top.size(); i++) {
					const Dtype loss_weight = m_layer_param.loss_weight(i);
					if (loss_weight == Dtype(0))
						continue;
					this->SetLoss(i, loss_weight);
					JaffeSet(top[i]->GetCount(), loss_weight, top[i]->GetMutableDiff());
				}
		}

		//hsz0409
		// 在不同的Layer会被覆盖成不同的版本
		virtual  void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
								 const vector<JBlob<Dtype>*>& top) {}
		virtual void Reshape(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) = 0;
		virtual inline const char* GetType() const {
			return "";
		}

		inline bool GetIsShared() const {
			return m_is_shared;
		}

		inline void SetShared(bool is_shared) {
			m_is_shared = is_shared;
		}

		vector<shared_ptr<JBlob<Dtype> > >& GetBlobs() {
			return m_blobs;
		}
		const LayerParameter& GetLayerParam() const {
			return m_layer_param;
		}
		inline Dtype GetLoss(const int top_index) const {
			return (m_loss.size() > top_index) ? m_loss[top_index] : Dtype(0);
		}
		inline void SetLoss(const int top_index, const Dtype value) {
			if (m_loss.size() <= top_index)
				m_loss.resize(top_index + 1, Dtype(0));
			m_loss[top_index] = value;
		}
		inline bool GetParamPropagateDown(const int param_id) {
			return (m_param_propagate_down.size() > param_id) ?
				m_param_propagate_down[param_id] : false;
		}
		inline void SetParamPropagateDown(const int param_id, const bool value) {
			if (m_param_propagate_down.size() > param_id)
				m_param_propagate_down[param_id] = value;
			else
				m_param_propagate_down.resize(param_id + 1, true);
		}
		Dtype LayerForward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top);
		// 如果返回 true, Init() 将会为 layer 生成匿名 blob 来满足最少的 top 需求
		virtual inline bool AutoTopBlobs() const {
			return false;
		} 
		// @override 指定 layer 最少的 top 数量
		virtual inline int MinTopBlobs() const {
			return -1;
		}

		// @override 指定 layer 最多的 top 数量
		virtual inline int MaxTopBlobs () const {
			return -1;
		}
		// @override 制定 layer top blob 数量
		virtual inline int ExactNumTopBlobs() const {
			return -1;
		}
		// @override 指定 layer 最少的 bottom 数量
		virtual inline int MinBottomBlobs() const {
			return -1;
		}

		// @override 指定 layer 最多的 bottom 数量
		virtual inline int MaxBottomBlobs () const {
			return -1;
		}
		// @override 制定 layer bottom blob 数量
		virtual inline int ExactNumBottomBlobs() const {
			return -1;
		}
		// 在数据并行计算中，指定 layer 是否被共享
		// 默认情况中，除了 data layer，一般 layer 都不该共享
		virtual inline bool ShareInParallel() const {
			return false;
		}	

	protected:
		LayerParameter m_layer_param;

       	vector<shared_ptr<JBlob<Dtype>>> m_blobs;
		vector<bool> m_param_propagate_down;

		vector<Dtype> m_loss;
		virtual void Forward(const vector<JBlob<Dtype>*>& bottom,
			const vector<JBlob<Dtype>*>& top) = 0;
		// 如果没有显卡，则运行CPU版本的Forward()
		virtual void ForwardGpu(const vector<JBlob<Dtype>*>& bottom,
						const vector<JBlob<Dtype>*>& top) {
			return Forward(bottom, top);
		}

	private:
		bool m_is_shared;
		
	}; // class JLayer
} // namespace jaffe
#endif
