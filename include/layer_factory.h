#ifndef LAYER_FACTORY_H_H
#define LAYER_FACTORY_H_H

#include <iostream>
#include <map>

#include "layer.h"

using std::cout;
using std::endl;
using std::map;

namespace jaffe {
	
	template <typename Dtype>
	class JLayer;

	template <typename Dtype>
	class JLayerRegistry {
	public:
		typedef shared_ptr<JLayer<Dtype> > (*Creator)(const LayerParameter&);
		typedef std::map<string, Creator> CreatorRegistry;

		static CreatorRegistry& Registry() {
			static CreatorRegistry* c_registry = new CreatorRegistry;
			return *c_registry;
		}
		
		static shared_ptr<JLayer<Dtype> > CreateLayer(const LayerParameter& param) {
			const string type = param.type();
			//cout << "type = " << type << endl;
			CreatorRegistry& registry = Registry();
			//cout << registry.size() << endl;
			return registry[type](param);
		}

		static void AddCreator(const string& type, Creator creator) {
			CreatorRegistry& registry = Registry();
			registry[type] = creator;
		}

	private:
		JLayerRegistry() {}

	}; // class JLayerRegistry

	template <typename Dtype>
	class JLayerRegisterer {
	public:
		JLayerRegisterer(const string& type,
			shared_ptr<JLayer<Dtype> > (*creator)(const LayerParameter&)) {
			JLayerRegistry<Dtype>::AddCreator(type, creator);
		}		
	}; // class JLayerRegisterer

#define REGISTER_LAYER_CREATOR(type, creator)	\
	static JLayerRegisterer<float> g_creator_f_##type(#type, creator<float>);	\
	static JLayerRegisterer<double> g_creator_d_##type(#type, creator<double>)	\

#define REGISTER_LAYER_CLASS(type)	\
	template <typename Dtype>	\
	shared_ptr<JLayer<Dtype> > Creator_##type##Layer(const LayerParameter& param)	\
	{	\
		return shared_ptr<JLayer<Dtype> > (new J##type##Layer<Dtype>(param));	\
	}	\
	REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
	
} // namespace jaffe
#endif
