// huangshize 2016.04.05
// 初步完成了JNet::Init()
// 不过其中的每一层参数的载入，以及每一层的SetUp还未实现
// huangshize 2016.04.09
// 更新了Net::Init()
// m_param->GetLayerParam存在bug

#include "net.h"

namespace jaffe {

    template <typename Dtype>
    JNet<Dtype>::JNet(const NetParameter& param, const JNet* root_net)
            : m_root_net(root_net) {
        Init(param);
    }

    template <typename Dtype>
    JNet<Dtype>::JNet(const string& param_file, Phase phase, const JNet* root_net)
            : m_root_net(root_net) {
        NetParameter param;
        ReadNetParamsFromTextFileOrDie(param_file, &param);
        //param.mutable_state()->set_phase(phase);
        Init(param);
    }

    template <typename Dtype>
    void JNet<Dtype>::Init(const NetParameter& origin_param){
        m_net_name = origin_param.name();
        m_phase = origin_param.state().phase();
        // ========  处理NetParameter  ========
        // 根据NetState来决定Layer是否引入到Net中
        NetParameter filtered_param;
        FilterNet(origin_param, &filtered_param);
        // 从filtered_param生成加入了splits层的NetParameter
        // util/insert_splits
        NetParameter param;
        InsertSplits(filtered_param, &param);
        // ========  建立所有layer并设置layer之间的连接  ========
        // 处理input
        map<string, int> blob_name_to_idx;
        set<string> available_blobs;
        m_memory_used = 0;
        // 不同层的input和output
        m_bottom.resize(param.layer_size());
        m_top.resize(param.layer_size());
        m_bottom_id.resize(param.layer_size());
        m_top_id.resize(param.layer_size());
        m_bottom_need_backward.resize(param.layer_size());
        m_param_id_vecs.resize(param.layer_size());

        // ========  迭代的初始化每一层layer  ========
        for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
            // 一般情况下root_net_为空
            // 一般情况下 share_from_root 为 false
            bool share_from_root = false;
            if (m_root_net) {
                //share_from_root = !Jaffe::root_solver()
                                  //&& m_root_net->m_layers[layer_id]->ShareInParallel();
			}
            // 设置每一层layer的phase
            //if (!param.layer(layer_id).has_phase()) {
            //param.mutable_layer(layer_id)->set_phase(m_phase);
            //}
            // layer ========  设置layer  ========
            const LayerParameter &layer_param = param.layer(layer_id);
            //if (layer_param.propagate_down_size() > 0) {
//				CHECK_EQ(layer_param.propagate_down_size(),
//						 layer_param.bottom_size())
//				<< "propagate_down param must be specified "
//				<< "either 0 or bottom_size times ";
            //}
            // layer ========  利用工厂类新建layer对象  ========
            if (share_from_root) {
//				LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
                m_layers.push_back(m_root_net->m_layers[layer_id]);
                m_layers[layer_id]->SetShared(true);
            } else {
                m_layers.push_back(JLayerRegistry<Dtype>::CreateLayer(layer_param));
            }
            m_layer_names.push_back(layer_param.name());
//			LOG_IF(INFO, Caffe::root_solver())
//			<< "Creating Layer " << layer_param.name();
            //bool need_backward = false;
//
//			// layer ========  计算出当前该层的输入和输出blob  ========
            // bottom
            for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id) {
                AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
//				const int blob_id = AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
//				// 设置need_backward，不过暂时忽略
                //need_backward |= blob_need_backward_[blob_id];
            }
            // top
            int num_top = layer_param.top_size();
            for (int top_id = 0; top_id < num_top; ++top_id) {
                AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
//				// 将InputLayer的top设置为net的input
                if (layer_param.type() == "Input") {
                    const int blob_id = m_blobs.size() - 1;
                    m_net_input_blob_indices.push_back(blob_id);
                    m_net_input_blobs.push_back(m_blobs[blob_id].get());
                }
            }
            // 如果layer的AutoTopBlobs()为true，并且LayerParameter指定的blob数量少于需要的数量，进行调整
            // 需要的blobs数量由 ExactNumTopBlobs() 或 MinTopBlobs() 指定
            JLayer<Dtype>* layer = m_layers[layer_id].get();
            if (layer->AutoTopBlobs()) {
                const int needed_num_top =
                        std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
                for (; num_top < needed_num_top; ++num_top) {
                    // 添加的top都是匿名对象，此时不改变 available_blobs 或 blob_name_to_idx
                    AppendTop(param, layer_id, num_top, NULL, NULL);
                }
            }
//			// layer ========  当该层被网络连接之后，开始设置该层  ========
            if (share_from_root) {
//				// 如果使用 root_net（已经配置好连接了）， 要进行配置
                const vector<JBlob<Dtype> *> &base_top = m_root_net->m_top[layer_id];
                const vector<JBlob<Dtype> *> &this_top = this->m_top[layer_id];
                for (int top_id = 0; top_id < base_top.size(); ++top_id) {
                    this_top[top_id]->ReshapeLike(*base_top[top_id]);
//					LOG(INFO) << "Created top blob " << top_id << " (shape: "
//					<< this_top[top_id]->shape_string() <<  ") for shared layer "
//					<< layer_param.name();
                }
            } else {
				//cout << "SetUp Layer " << layer_id << ")\t" 
					//<< m_layers[layer_id]->GetType();
                m_layers[layer_id]->SetUp(m_bottom[layer_id], m_top[layer_id]);
				//cout << "\t\tDone" << endl;
            }
//			LOG_IF(INFO, Caffe::root_solver())
//			<< "Setting up " << layer_names_[layer_id];
            for (int top_id = 0; top_id < m_top[layer_id].size(); ++top_id) {
                if (m_blob_loss_weights.size() <= m_top_id[layer_id][top_id])
                    m_blob_loss_weights.resize(m_top_id[layer_id][top_id] + 1,
                                               Dtype(0));
                m_blob_loss_weights[m_top_id[layer_id][top_id]] =
                        layer->GetLoss(top_id);
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
//				if (layer->GetLoss(top_id)) {
//					LOG_IF(INFO, Caffe::root_solver())
//					<< "    with loss weight " << layer->loss(top_id);
//				}
                m_memory_used += m_top[layer_id][top_id]->GetCount();
            }
//			LOG_IF(INFO, Caffe::root_solver())
//			<< "Memory required for data: " << memory_used_ * sizeof(Dtype);
            const int param_size = layer_param.param_size();
            const int num_param_blobs = m_layers[layer_id]->GetBlobs().size();
//			CHECK_LE(param_size, num_param_blobs)
//			<< "Too many params specified for layer " << layer_param.name();
            ParamSpec default_param_spec;

            for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
                const ParamSpec* param_spec;
                if(param_id < param_size)
                    param_spec = &layer_param.param(param_id);
                else
                    param_spec = &default_param_spec;
                //const bool param_need_backward = param_spec->lr_mult() != 0;
                //need_backward |= param_need_backward;
                //m_layers[layer_id]->SetParamPropagateDown(param_id,
                //param_need_backward);
            }
            for (int param_id = 0; param_id < num_param_blobs; ++param_id)
                AppendParam(param, layer_id, param_id);
            // layer ========  设置是否需要 backward  ========
            // 除了 input layer，一般层 need_backward 为 true， 后来根据需要可设为false
            //m_layer_need_backward.push_back(need_backward);
            //if (need_backward) {
            //for (int top_id = 0; top_id < m_top_id[layer_id].size(); ++top_id) {
            //m_blob_need_backward[m_top_id[layer_id][top_id]] = true;
            //}
            //}
        }

        for (set<string>::iterator it = available_blobs.begin();
             it != available_blobs.end(); ++it) {
            m_net_output_blobs.push_back(m_blobs[blob_name_to_idx[*it]].get());
            m_net_output_blob_indices.push_back(blob_name_to_idx[*it]);
        }
        for (size_t blob_id = 0; blob_id < m_blob_names.size(); ++blob_id) {
            m_blob_name_id[m_blob_names[blob_id]] = blob_id;
        }
        for (size_t layer_id = 0; layer_id < m_layer_names.size(); ++layer_id) {
            m_layer_name_id[m_layer_names[layer_id]] = layer_id;
        }
        ShareWeights();
		//cout << endl;
    }

    template <typename Dtype>
    const vector<JBlob<Dtype>*>& JNet<Dtype>::NetForward(Dtype* loss) {
        if (loss != NULL)
            *loss = NetForwardFromTo(0, m_layers.size() - 1);
        else
            NetForwardFromTo(0, m_layers.size() - 1);
        return m_net_output_blobs;
    }

    template <typename Dtype>
    Dtype JNet<Dtype>::NetForwardFromTo(const int start, const int end) {
        Dtype loss = 0;
        for (int i = start; i <= end; i++) {
			//cout << i << ")\t" << m_layers[i]->GetType() << endl;
            Dtype layer_loss = m_layers[i]->LayerForward(m_bottom[i], m_top[i]);
        }
        return loss;
    }


    template <typename Dtype>
    void JNet<Dtype>::ShareWeights() {
        for (int i = 0; i < m_params.size(); ++i) {
            if (m_param_owners[i] < 0) { continue; }
            m_params[i]->ShareData(*m_params[m_param_owners[i]]);
            m_params[i]->ShareDiff(*m_params[m_param_owners[i]]);
        }
    }

    // copy
    template <typename Dtype>
    void JNet<Dtype>::FilterNet(const NetParameter& param, NetParameter* param_filtered) {
        NetState net_state(param.state());
        param_filtered->CopyFrom(param);
        param_filtered->clear_layer();
        for (int i = 0; i < param.layer_size(); ++i) {
            const LayerParameter& layer_param = param.layer(i);
            const string& layer_name = layer_param.name();
            //CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
            //<< "Specify either include rules or exclude rules; not both.";

            // 默认情况下，layer是会被引入到net中，include这一属性一般缺省
            bool layer_included = (layer_param.include_size() == 0);
            for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
                if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
                    layer_included = false;
                }
            }
            for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
                if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
                    layer_included = true;
                }
            }
            if (layer_included) {
                param_filtered->add_layer()->CopyFrom(layer_param);
            }
        }
    }
    // copy
    template <typename Dtype>
    bool JNet<Dtype>::StateMeetsRule(const NetState& state,
                                     const NetStateRule& rule, const string& layer_name) {
        // Check whether the rule is broken due to phase.
        if (rule.has_phase()) {
            if (rule.phase() != state.phase()) {
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "The NetState phase (" << state.phase()
//				<< ") differed from the phase (" << rule.phase()
//				<< ") specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to min level.
        if (rule.has_min_level()) {
            if (state.level() < rule.min_level()) {
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "The NetState level (" << state.level()
//				<< ") is above the min_level (" << rule.min_level()
//				<< ") specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to max level.
        if (rule.has_max_level()) {
            if (state.level() > rule.max_level()) {
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "The NetState level (" << state.level()
//				<< ") is above the max_level (" << rule.max_level()
//				<< ") specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to stage. The NetState must
        // contain ALL of the rule's stages to meet it.
        for (int i = 0; i < rule.stage_size(); ++i) {
            // Check that the NetState contains the rule's ith stage.
            bool has_stage = false;
            for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
                if (rule.stage(i) == state.stage(j)) { has_stage = true; }
            }
            if (!has_stage) {
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "The NetState did not contain stage '" << rule.stage(i)
//				<< "' specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to not_stage. The NetState must
        // contain NONE of the rule's not_stages to meet it.
        for (int i = 0; i < rule.not_stage_size(); ++i) {
            // Check that the NetState contains the rule's ith not_stage.
            bool has_stage = false;
            for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
                if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
            }
            if (has_stage) {
//				LOG_IF(INFO, Caffe::root_solver())
//				<< "The NetState contained a not_stage '" << rule.not_stage(i)
//				<< "' specified by a rule in layer " << layer_name;
                return false;
            }
        }
        return true;
    }

    // 增加一个新的top
    template <typename Dtype>
    void JNet<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                                const int top_id, set<string>* available_blobs,
                                map<string, int>* blob_name_to_idx)
    {
        shared_ptr<LayerParameter> layer_param(new LayerParameter(param.layer(layer_id)));
        const string& blob_name = (layer_param->top_size() > top_id) ?
                                  layer_param->top(top_id) : "(automatic)";
        // Check if we are doing in-place computation
        if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
            blob_name == layer_param->bottom(top_id)) {
            // In-place computation
//			LOG_IF(INFO, Caffe::root_solver())
//			<< layer_param->name() << " -> " << blob_name << " (in-place)";
            m_top[layer_id].push_back(m_blobs[(*blob_name_to_idx)[blob_name]].get());
            m_top_id[layer_id].push_back((*blob_name_to_idx)[blob_name]);
        } else if (blob_name_to_idx &&
                   blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
            // If we are not doing in-place computation but have duplicated blobs,
            // raise an error.
//			LOG(FATAL) << "Top blob '" << blob_name
//			<< "' produced by multiple sources.";
        } else {
            // Normal output.
//			if (Caffe::root_solver()) {
//				LOG(INFO) << layer_param->name() << " -> " << blob_name;
//			}
            shared_ptr<JBlob<Dtype> > blob_pointer(new JBlob<Dtype>());
            const int blob_id = m_blobs.size();
            m_blobs.push_back(blob_pointer);
            m_blob_names.push_back(blob_name);
            m_blob_need_backward.push_back(false);
            if (blob_name_to_idx)
                (*blob_name_to_idx)[blob_name] = blob_id;
            m_top_id[layer_id].push_back(blob_id);
            m_top[layer_id].push_back(blob_pointer.get());
        }
        if (available_blobs)
            available_blobs->insert(blob_name);
    }

    // 增加一个新的bottom
    template <typename Dtype>
    int JNet<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
                                  const int bottom_id, set<string>* available_blobs,
                                  map<string, int>* blob_name_to_idx)
    {
        const LayerParameter& layer_param = param.layer(layer_id);
        const string& blob_name = layer_param.bottom(bottom_id);
        //if (available_blobs->find(blob_name) == available_blobs->end()) {
//			LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
//			<< layer_param.name() << "', bottom index " << bottom_id << ")";
        //}
        const int blob_id = (*blob_name_to_idx)[blob_name];
        m_bottom[layer_id].push_back(m_blobs[blob_id].get());
        m_bottom_id[layer_id].push_back(blob_id);
        available_blobs->erase(blob_name);
        bool need_backward = m_blob_need_backward[blob_id];
        // 在 TEST 情况下， layer_param.propagate_down_size() 为 0
        if (layer_param.propagate_down_size() > 0) {
            need_backward = layer_param.propagate_down(bottom_id);
        }
        m_bottom_need_backward[layer_id].push_back(need_backward);
        return blob_id;
    }

    template <typename Dtype>
    void JNet<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                                  const int param_id) {
        const LayerParameter& layer_param = m_layers[layer_id]->GetLayerParam();
        const int param_size = layer_param.param_size();
        string param_name =
                (param_size > param_id) ? layer_param.param(param_id).name() : "";
        if (param_name.size()) {
            m_param_display_names.push_back(param_name);
        } else {
            ostringstream param_display_name;
            param_display_name << param_id;
            m_param_display_names.push_back(param_display_name.str());
        }
        const int net_param_id = m_params.size();
        m_params.push_back(m_layers[layer_id]->GetBlobs()[param_id]);
        m_param_id_vecs[layer_id].push_back(net_param_id);
        m_param_layer_indices.push_back(make_pair(layer_id, param_id));
        ParamSpec default_param_spec;
        const ParamSpec* param_spec;
        if(layer_param.param_size() > param_id)
            param_spec = &layer_param.param(param_id);
        else
            param_spec = &default_param_spec;
        if (!param_size || !param_name.size() || (param_name.size() &&
                                                  m_param_names_index.find(param_name) == m_param_names_index.end())) {
            // This layer "owns" this parameter blob -- it is either anonymous
            // (i.e., not given a param_name) or explicitly given a name that we
            // haven't already seen.
            m_param_owners.push_back(-1);
            if (param_name.size())
                m_param_names_index[param_name] = net_param_id;
            const int learnable_param_id = m_learnable_params.size();
            m_learnable_params.push_back(m_params[net_param_id].get());
            m_learnable_param_ids.push_back(learnable_param_id);
            m_has_params_lr.push_back(param_spec->has_lr_mult());
            m_has_params_decay.push_back(param_spec->has_decay_mult());
            m_params_lr.push_back(param_spec->lr_mult());
            m_params_weight_decay.push_back(param_spec->decay_mult());
        } else {
            // Named param blob with name we've seen before: share params
            const int owner_net_param_id = m_param_names_index[param_name];
            m_param_owners.push_back(owner_net_param_id);
            const pair<int, int>& owner_index =
                    m_param_layer_indices[owner_net_param_id];
            const int owner_layer_id = owner_index.first;
            const int owner_param_id = owner_index.second;
//			LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
//			<< "' owned by "
//			<< "layer '" << layer_names_[owner_layer_id] << "', param "
//			<< "index " << owner_param_id;
            JBlob<Dtype>* this_blob = m_layers[layer_id]->GetBlobs()[param_id].get();
            JBlob<Dtype>* owner_blob =
                    m_layers[owner_layer_id]->GetBlobs()[owner_param_id].get();
            const int param_size = layer_param.param_size();
            if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                          ParamSpec_DimCheckMode_PERMISSIVE)) {
                // Permissive dimension checking -- only check counts are the same.
//				CHECK_EQ(this_blob->count(), owner_blob->count())
//				<< "Cannot share param '" << param_name << "' owned by layer '"
//				<< layer_names_[owner_layer_id] << "' with layer '"
//				<< layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
//				<< "shape is " << owner_blob->shape_string() << "; sharing layer "
//				<< "shape is " << this_blob->shape_string();
            } else {
                // Strict dimension checking -- all dims must be the same.
//				CHECK(this_blob->shape() == owner_blob->shape())
//				<< "Cannot share param '" << param_name << "' owned by layer '"
//				<< layer_names_[owner_layer_id] << "' with layer '"
//				<< layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
//				<< "shape is " << owner_blob->shape_string() << "; sharing layer "
//				<< "expects shape " << this_blob->shape_string();
            }
            const int learnable_param_id = m_learnable_param_ids[owner_net_param_id];
            m_learnable_param_ids.push_back(learnable_param_id);
            if (param_spec->has_lr_mult()) {
                if (m_has_params_lr[learnable_param_id]) {
//					CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
//					<< "Shared param '" << param_name << "' has mismatched lr_mult.";
                } else {
                    m_has_params_lr[learnable_param_id] = true;
                    m_params_lr[learnable_param_id] = param_spec->lr_mult();
                }
            }
            if (param_spec->has_decay_mult()) {
                if (m_has_params_decay[learnable_param_id]) {
//					CHECK_EQ(param_spec->decay_mult(),
//							 params_weight_decay_[learnable_param_id])
//					<< "Shared param '" << param_name << "' has mismatched decay_mult.";
                } else {
                    m_has_params_decay[learnable_param_id] = true;
                    m_params_weight_decay[learnable_param_id] = param_spec->decay_mult();
                }
            }
        }
    }

    template <typename Dtype>
    void JNet<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
        if (trained_filename.size() >= 3 &&
            trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0)
            CopyTrainedLayersFromHDF5(trained_filename);
        else
            CopyTrainedLayersFromBinaryProto(trained_filename);
    }

    template <typename Dtype>
    void JNet<Dtype>::CopyTrainedLayersFromBinaryProto(const string trained_filename) {
        NetParameter param;
        ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
        CopyTrainedLayersFrom(param);
    }

    template <typename Dtype>
    void JNet<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
        int num_layers = param.layer_size();
        for (int i = 0; i < num_layers; i++) {
            const LayerParameter& layer_param = param.layer(i);
            const string& layer_name = layer_param.name();
            int layer_id = 0;
            while (layer_id != m_layer_names.size() &&
                   m_layer_names[layer_id] != layer_name)
                layer_id ++;
            if (layer_id == m_layer_names.size())
                continue;
            vector<shared_ptr<JBlob<Dtype> > >& target_blobs =
                    m_layers[layer_id]->GetBlobs();
            for (int j = 0; j < target_blobs.size(); j++) {
                if (!target_blobs[j]->ShapeEquals(layer_param.blobs(j))) {
                    JBlob<Dtype> source_blob;
                    const bool kReshape = true;
                    source_blob.FromProto(layer_param.blobs(j), kReshape);
                }
                const bool kReshape = false;
                target_blobs[j]->FromProto(layer_param.blobs(j), kReshape);
            }
        }
    }

    template <typename Dtype>
    void JNet<Dtype>::Reshape() {
        for (int i = 0; i < m_layers.size(); i++) {

            m_layers[i]->Reshape(m_bottom[i], m_top[i]);

			/*
			for (int j = 0; j < m_bottom[i].size(); j++) {
				cout << "bottom[" << i << ", " << j << "] shape is " << 
					m_bottom[i][j]->GetShapeString() << endl;
			}
			for (int j = 0; j < m_top[i].size(); j++) {
				cout << "top[" << i << ", " << j << "] shape is " << 
					m_top[i][j]->GetShapeString() << endl;
			}
			*/
        }
    }

    template <typename Dtype>
    const shared_ptr<JBlob<Dtype > > JNet<Dtype>::GetBlobByName(const string&
    blob_name) const {
        shared_ptr<JBlob<Dtype> > blob_ptr;
        if (HasBlob(blob_name)) {
            blob_ptr = m_blobs[m_blob_name_id.find(blob_name)->second];
        }
        else {
            blob_ptr.reset((JBlob<Dtype>*)(NULL));
        }
        return blob_ptr;
    }

    template <typename Dtype>
    bool JNet<Dtype>::HasBlob(const string& blob_name) const {
        return m_blob_name_id.find(blob_name) != m_blob_name_id.end();
    }

    template class JNet<float>;
    template class JNet<double>;
} // namespace jaffe
