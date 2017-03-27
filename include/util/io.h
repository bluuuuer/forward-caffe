// huangshize 2016.03.16
// huangshize 2016.04.01
// 增加了多个ReadImageToDatum函数接口
#ifndef IO_H_H
#define IO_H_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "blob.h"

using std::fstream;
using std::ios;
using std::string;
using std::vector;

namespace jaffe {
	
	using ::google::protobuf::Message;
	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;

	bool ReadProtoFromTextFile(const char* filename, Message* proto);
	
	inline bool ReadProtoFromTextFile(const string& filename , Message* proto) {
		return ReadProtoFromTextFile(filename.c_str(), proto);
	}

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

	inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
		return ReadProtoFromBinaryFile(filename.c_str(), proto);
	}

	bool ReadFileToDatum(const string& filename, const int label, Datum* datum);
	//  参考caffe，该方法用于读图并返回一个Mat数据
	cv::Mat ReadImageToCVMat(const string& filename, const int height = 0, const int width = 0,
						 const bool is_color = 1);

	void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

	static bool matchExt(const string& fn, string en);

	bool ReadImageToDatum(const string& filename, const int label, const int height,
					  const int width, const bool is_color, const string& encoding, Datum* datum);

	// hsz encoded默认是空串
	inline bool ReadImageToDatum(const string& filename, const int label,
		const int height, const int width, const bool is_color, Datum* datum) {
		return ReadImageToDatum(filename, label, height, width, is_color, "", datum);
	}

	inline bool ReadImageToDatum(const string& filename, const int label,
		const int height, const int width, Datum* datum) {
		return ReadImageToDatum(filename, label, height, width, true, datum);
	}

	inline bool ReadImageToDatum(const string& filename, const int label,
		const bool is_color, Datum* datum) {
		return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
	}

	inline bool ReadImageToDatum(const string& filename, const int label, Datum* datum) {
		return ReadImageToDatum(filename, label, 0, 0, true, datum);
	}

	inline bool ReadImageToDatum(const string& filename, const int label,
		const std::string & encoding, Datum* datum) {
		return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
	}

	void PutDatumToDataset(Datum* datum, const string& db_backend);
	
	inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
		ReadProtoFromBinaryFile(filename, proto);
	}

	inline void ReadProtoFromBinaryFileOrDie(const string& filename, Message* proto) {
		ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
	}

	template <typename Dtype>
	void ImageDataToBlob(JBlob<Dtype>* blob, const int img_id, const cv::Mat& image);

	template <typename Dtype>
	void BlobToGreyImage(const JBlob<Dtype>* blob, const Dtype scale, cv::Mat* imge);
	
	
	template <typename Dtype>
	cv::Mat_<Dtype> GetAffineImage(const cv::Mat& src, cv::Mat& dst, 
		const vector<Dtype>& landmark, const vector<int> center_ind, 
		const int norm_mode = 1, const float norm_ratio = 0.5, 
		const bool fill_type = true, const int value = 0, const int im_height = 40,
		const int im_width = 40, const bool im_is_color = true);

	template <typename Dtype>
	cv::Point_<Dtype> GetAffineImage_GetSrcCenter(const vector<Dtype>& landmark,
		const vector<int> center_ind);

	template <typename Dtype>
	Dtype GetAffineImage_GetScale(const cv::Mat& src, cv::Mat& dst,
		const vector<Dtype>& landmark, const int center_mode, const float norm_ratio);

	template <typename Dtype>
	Dtype GetAffineImage_GetAngle(const vector<Dtype>& landmark);

	template <typename Dtype>
	cv::Mat_<Dtype> Get_Affine_matrix(const cv::Point_<Dtype>& srcCenter,
		const cv::Point_<Dtype>& dstCenter, const Dtype alpha, const Dtype scale);

	template <typename Dtype>
	void mAffineWarp(const cv::Mat_<Dtype>& M, const cv::Mat& srcImg, cv::Mat& dstImg,
		const bool fill_type, const uchar value);

} // namespace jaffe
#endif
