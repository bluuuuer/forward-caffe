// xw
// huangshize 2016.04.01
// 写入了class JImageDataLayer，未测试

#ifndef DATA_LAYERS_H_H
#define DATA_LAYERS_H_H

#include "layer.h"
#include "data_param.h"

namespace jaffe {
	template <typename Dtype>
	class JBaseDataLayer : public JLayer<Dtype>{
	public:
		JBaseDataLayer(){};
		JBaseDataLayer(const JLayerParam& param);  // hsz0401 读入的参数需要考证
		~JBaseDataLayer(){};
		//virtual void DataLayerSetUp(const vector<JBlob<Dtype>*>& bottom,
		//	const vector<JBlob<Dtype>*>& top){}
	protected:
		//TransformationParameter transform_param;
		//vector<DataTransformer*> data_transformer;
	};

	template <typename Dtype>
	class JBaseFetchingDataLayer : public JBaseDataLayer<Dtype>{
	public:
		JBaseFetchingDataLayer(){};
		~JBaseFetchingDataLayer(){};

		// hsz0401 说是说执行 DataLayerSetUp(...)
//		void LayerSetUp(const vector<JBlob<Dtype>*>& bottom,
//			const vector<JBlob<Dtype>*>& top);
//		virtual void Forward(const vector<Blob<Dtype>*>& bottom,
//							 const vector<Blob<Dtype>*>& top);


	protected:
//		JBlob<Dtype> m_transformed_data;
	};

	template <typename Dtype>
	class JDataLayer : public  JBaseFetchingDataLayer<Dtype>{
	public:
		JDataLayer(){
			m_param = new JDataParam;
		};
		~JDataLayer(){
			delete m_param;
		};

		bool Init(const vector<string> param);
		virtual bool Show();

	private:
		JDataParam* m_param;

	}; // class JDataLayer

	// hsz0401  用于读取Image的DataLayer，注意继承
	// 
	//template <typename Dtype>
	//class JImageDataLayer : public JBaseFetchingDataLayer<Dtype>{
	//public:
	//	explicit JImageDataLayer(const JLayerParam& param)
	//		: JBaseFetchingDataLayer(const JLayerParam& param) {}
	//	virtual ~JImageDataLayer();

	//	virtual void DataLayerSetUp(const vector<JBlob<Dtype>*>& bottom,
	//		const vector<JBlob<Dtype>*>& top);

	//	virtual inline const char* type() const { return "ImageData"; }
	//	virtual inline int ExactNumBottomBlobs() const { return 0; }
	//	virtual inline int ExactNumTopBlobs() const { return 2; }

	//	void FetchData();

	//protected:

	//	// 考虑覆盖JBaseFetchingDataLayer的同名函数
	//	virtual void Forward(const vector<Blob<Dtype>*>& bottom,
	//		const vector<Blob<Dtype>*>& top);

	//	vector<std::pair<std::string, int> > m_lines;
	//	int m_lines_id;

	//	//
	//	JBlob<Dtype> m_data;
	//	JBlob<Dtype> m_label;
	//};

}
#endif