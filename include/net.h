// huangshzie 0406
// 修改了了部分成员和Init()
#ifndef NET_H_H
#define NET_H_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>

#include "layer.h"
#include "util/upgrade_proto.hpp"
#include "util/insert_splits.h"


using std::string;
using std::cout;
using std::endl;

namespace jaffe {

	template <typename Dtype>
	class JNet{
	public:
//		JNet() {};
		explicit JNet(const NetParameter& param, const JNet* root_net = NULL);
		explicit JNet(const string& param_file, Phase phase = TEST, 
			const JNet* root_net = NULL);
		~JNet(){
		};

		// 初始化整个net
		void Init(const NetParameter& param);
		const vector<JBlob<Dtype>*>& NetForward(Dtype* loss = NULL);
		Dtype NetForwardFromTo(const int start, const int end);
		// ===================================================
		// 根据NetState状态，来删除某些层
		static void FilterNet(const NetParameter& param, NetParameter* param_filtered);
		// 配合FilterNet，判断NetState是否符合NetStateRule的规则
		static bool StateMeetsRule(const NetState& state, const NetStateRule& rule, const string& layer_name);
		// ===================================================
		void ShareWeights();
		void Reshape();

		void CopyTrainedLayersFrom(const string trained_filename);
		void CopyTrainedLayersFromHDF5(const string trained_filename) {} // unfinished
		void CopyTrainedLayersFromBinaryProto(const string trained_filename);
		void CopyTrainedLayersFrom(const NetParameter& param);

		inline const vector<JBlob<Dtype>*>& GetInputBlobs() const {
			return m_net_input_blobs;
		}
		inline const vector<JBlob<Dtype>*>& GetOutputBlobs() const {
			return m_net_output_blobs;
		}
		const shared_ptr<JBlob<Dtype> > GetBlobByName(const string& blob_name) const;
		bool HasBlob(const string& blob_name) const;
		inline const vector<shared_ptr<JLayer<Dtype> > >& GetLayers() const {
				return m_layers;
		}
		inline const vector<JBlob<Dtype>*> GetBottomBlobsByIndex(const int i) {
			int index = i;
			if (index < 0) {
				index += m_layers.size();
			}
			return m_bottom[index];
		}
		inline const vector<JBlob<Dtype>*> GetTopBlobsByIndex(const int i) {
			int index = i;
			if (index < 0) {
				index += m_layers.size();
			}
			return m_top[index];
		}

	protected:
		// 增加新的top blob到net中
		void AppendTop(const NetParameter& param, const int layer_id,
					   const int top_id, set<string>* available_blobs,
					   map<string, int>* blob_name_to_idx);
		// 增加新的bottom blob到net中
		int AppendBottom(const NetParameter& param, const int layer_id,
						 const int bottom_id, set<string>* available_blobs,
						 map<string, int>* blob_name_to_idx);
		// 增加新的parameter到net中
		void AppendParam(const NetParameter& param, const int layer_id,
						 const int param_id);
		// ===================================================
//		JNetParameter GetParameter(){ return *m_param; };
		// hsz0407
//		int GetLayerNum() { return  m_param->GetLayerNum(); }

		string m_net_name;
		Phase m_phase;
//		vector<JLayer<Dtype>*> m_layers;
		
		// hsz0402 layers name & layers id & map
		// 用来更好的管理每一个层
		vector<shared_ptr<JLayer<Dtype>>> m_layers;		// layers_
		vector<string> m_layer_names;		// layer_names_
		map<string, int> m_layer_name_id;		// layer_names_index_
		vector<bool> m_layer_need_backward;
		// hsz0402 用来管理所有的blobs
		vector<shared_ptr<JBlob<Dtype>>> m_blobs;		// blobs_
		vector<string> m_blob_names;		// blob_names_
		vector<int> m_blob_id;
		map<string, int> m_blob_name_id;	// blob_names_index_
		vector<bool> m_blob_need_backward;
		// hsz0402 用来管理所有的bolbs，分为bottom和top，注意储存的是指针
		vector<vector<JBlob<Dtype>*>> m_bottom;		// bottom_vecs_
		vector<vector<int>> m_bottom_id;	// bottom_id_vecs_
		vector<vector<bool> > m_bottom_need_backward;
		vector<vector<JBlob<Dtype>*>> m_top;	// top_vecs_
		vector<vector<int>> m_top_id;	//  top_id_vecs_
		//vector<vector<bool>> m_bottom_backword;
		// net 输入和输出的 blob
		vector<int> m_net_input_blob_indices;
		vector<int> m_net_output_blob_indices;
		vector<JBlob<Dtype>*> m_net_input_blobs;
		vector<JBlob<Dtype>*> m_net_output_blobs;

		/// Vector of weight in the loss (or objective) function of each net blob,
		/// indexed by blob_id.
		vector<Dtype> m_blob_loss_weights;
		vector<vector<int> > m_param_id_vecs;
		vector<int> m_param_owners;
		vector<string> m_param_display_names;
		vector<pair<int, int> > m_param_layer_indices;
		map<string, int> m_param_names_index;

		/// The parameters in the network.
		vector<shared_ptr<JBlob<Dtype> > > m_params;
		vector<JBlob<Dtype>*> m_learnable_params;
		/**
         * The mapping from params_ -> learnable_params_: we have
         * learnable_param_ids_.size() == params_.size(),
         * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
         * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
         * and learnable_params_[learnable_param_ids_[i]] gives its owner.
         */
		vector<int> m_learnable_param_ids;
		/// the learning rate multipliers for learnable_params_
		vector<float> m_params_lr;
		vector<bool> m_has_params_lr;
		/// the weight decay multipliers for learnable_params_
		vector<float> m_params_weight_decay;
		vector<bool> m_has_params_decay;
		// hsz0425
		// Net所使用的空间
		size_t m_memory_used;
		// The root net that actually holds the shared layers in data parallelism
		const JNet* m_root_net;


	};
}
#endif
