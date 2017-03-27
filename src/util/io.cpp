// huangshize 2016.03.30
// === io.hpp ===
// 提供用于解决数据输入输出的方法
// huangshize 2016.04.05
// 未能解决opencv库链接问题
#include "util/io.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fcntl.h> // open()
#include <unistd.h> // close()
#include <iostream>

#include <stdint.h>
#include <fstream>
#include <string>
#include <vector>

using std::cout;
using std::endl;

const int kProtoReadBytesLimit = INT_MAX;

namespace jaffe {
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    using google::protobuf::io::CodedInputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    using google::protobuf::io::CodedOutputStream;
    using google::protobuf::Message;

    bool ReadFileToDatum(const string& filename, const int label,
                         Datum* datum) {
        std::streampos size;

        fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
        if (file.is_open()) {
            size = file.tellg();
            std::string buffer(size, ' ');
            file.seekg(0, ios::beg);
            file.read(&buffer[0], size);
            file.close();
            datum->set_data(buffer);
            datum->set_label(label);
            datum->set_encoded(true);
            return true;
        } else {
            return false;
        }
    }
//  参考caffe，该方法用于读图并返回一个Mat数据

    cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width, const bool is_color)
    {
        cv::Mat cv_img;
        int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
        // if(!cv_img_origin.data)
        if (height > 0 && width > 0)
        {
            cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
        }
        else
        {
            cv_img = cv_img_origin;
        }
        return cv_img;
    }

    void CVMatToDatum(const cv::Mat& cv_img, Datum* datum)
    {
        // 图像的数据类型必须是未编码的
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->clear_data();
        datum->clear_float_data();
        datum->set_encoded(false);
        int datum_channels = datum->channels();
        int datum_height = datum->height();
        int datum_width = datum->width();
        int datum_size = datum_channels * datum_height * datum_width;
        std::string buffer(datum_size, ' ');
        for (int h = 0; h < datum_height; ++h) {
            const uchar* ptr = cv_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < datum_width; ++w) {
                for (int c = 0; c < datum_channels; ++c) {
                    int datum_index = (c * datum_height + h) * datum_width + w;
                    buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            }
        }
        datum->set_data(buffer);
    }
    static bool matchExt(const std::string & fn,
                         std::string en) {
        size_t p = fn.rfind('.');
        std::string ext = p != fn.npos ? fn.substr(p) : fn;
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        std::transform(en.begin(), en.end(), en.begin(), ::tolower);
        if ( ext == en )
            return true;
        if ( en == "jpg" && ext == "jpeg" )
            return true;
        return false;
    }

    bool ReadImageToDatum(const string& filename, const int label, const int height, const int width,
                          const bool is_color, const string& encoding, Datum* datum)
    {
        cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
        if (cv_img.data) {
            if (encoding.size()) {
                if ( (cv_img.channels() == 3) == is_color && !height && !width &&
                     matchExt(filename, encoding) )
                    return ReadFileToDatum(filename, label, datum);
                std::vector<uchar> buf;
                cv::imencode("."+encoding, cv_img, buf);
                datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                                            buf.size()));
                datum->set_label(label);
                datum->set_encoded(true);
                return true;
            }
            CVMatToDatum(cv_img, datum);
            datum->set_label(label);
            return true;
        } else {
            return false;
        }
    }


    bool ReadProtoFromTextFile(const char* filename, Message* proto) {
        int fd = open(filename, O_RDONLY);
        FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        NetParameter* p = static_cast<NetParameter* >(proto);
        delete input;
        close(fd);
        return success;
    }

    bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
        //	std::cout << "io.cpp->ReadProtoFromBinaryFile(" << filename << ", proto)" << std::endl;
        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            std::cout << "File not found" << filename << std::endl;
        }
        ZeroCopyInputStream* raw_input = new FileInputStream(fd);
        CodedInputStream* coded_input = new CodedInputStream(raw_input);
        coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

        bool success = proto->ParseFromCodedStream(coded_input);

        delete coded_input;
        delete raw_input;
        close(fd);
        return success;
    }

	template <typename Dtype>
	void ImageDataToBlob(JBlob<Dtype>* blob, const int img_id, const cv::Mat& image) {
		const int height = blob->GetHeight();
		const int width = blob->GetWidth();
		const int channels = blob->GetChannels();
		const int size = height * width;
		const int dims = channels * size;

		Dtype* data = blob->GetMutableData();
		int h_off = img_id * dims;
		for (int i = 0; i < height; i ++){
			const uchar* img_ptr = image.ptr<uchar>(i);
			int w_off = h_off;
			for (int j = 0; j < width; j++){
				int idex = w_off;
				for (int c = 0; c < channels; c++) {
					data[idex] = static_cast<Dtype>(*img_ptr++);
					idex += size;
				}
				w_off ++;
			}
			h_off += width;
		}
	}
	template void ImageDataToBlob(JBlob<float>* blob, const int img_id, 
		const cv::Mat& image);
	template void ImageDataToBlob(JBlob<double>* blob, const int img_id,
		const cv::Mat& image);

	template <typename Dtype>
	void BlobToGreyImage(const JBlob<Dtype>* blob, const Dtype scale,
			cv::Mat* image) {
		const Dtype* data = blob->GetData();
		for (int h = 0; h < image->rows; h++) {
			for (int w = 0; w < image->cols; w++) {
				int index = h * image->cols + w;
				Dtype temp = data[index] * scale;
				//cout << "blob->DataAt(1, 1, " <<  h << ", " << w <<") = " << 
					//data[index];
				//cout << "\ttemp = " << temp << endl;
				image->at<uchar>(h, w) = cv::saturate_cast<uchar>(temp);
			}
		}
		cout << "Done BlobToGreyImage()" << endl;
	}
	template void BlobToGreyImage(const JBlob<float>* blob, const float scale,
			cv::Mat* image);
	template void BlobToGreyImage(const JBlob<double>* blob, const double scale,
			cv::Mat* image);

	template <typename Dtype>
	cv::Mat_<Dtype> GetAffineImage(const cv::Mat& src, cv::Mat& dst,
			const vector<Dtype>& landmark, const vector<int> center_ind, 
			const int norm_mode, const float norm_ratio, const bool fill_type, 
			const int value, const int im_height, const int im_width, 
			const bool im_is_color) {
		const cv::Point_<Dtype> src_center = GetAffineImage_GetSrcCenter(landmark,
			center_ind);
		const cv::Point_<Dtype> dst_center(dst.cols / 2, dst.rows / 2);
		const Dtype scale = GetAffineImage_GetScale(src, dst, landmark, norm_mode,
			norm_ratio);
		const Dtype angle = GetAffineImage_GetAngle(landmark);
		const cv::Mat_<Dtype> affine_mat = Get_Affine_matrix(src_center, dst_center,
			-angle, scale);
		mAffineWarp(affine_mat, src, dst, fill_type, value);
		return affine_mat;
	}
	template cv::Mat_<float> GetAffineImage(const cv::Mat& src, cv::Mat& dst, 
		const vector<float>& landmark, const vector<int> center_ind,
		const int norm_mode, const float norm_ratio, const bool fille_type, 
		const int value, const int im_height, const int im_width, 
		const bool im_is_color);
	template cv::Mat_<double> GetAffineImage(const cv::Mat& src, cv::Mat& dst, 
		const vector<double>& landmark, const vector<int> center_ind,
		const int norm_mode, const float norm_ratio, const bool fille_type, 
		const int value, const int im_height, const int im_width, 
		const bool im_is_color);

	template <typename Dtype>
	cv::Point_<Dtype> GetAffineImage_GetSrcCenter(const vector<Dtype>& landmark,
    		const vector<int> center_ind) {
  		cv::Point_<Dtype> src_center;
  		if (center_ind.size() == 0 ||
    		(center_ind.size() == 1 && center_ind[0] == -1)) {
    		const Dtype left_eye_x = landmark[0];
    		const Dtype left_eye_y = landmark[1];
    		const Dtype right_eye_x = landmark[2];
    		const Dtype right_eye_y = landmark[3];
    		const Dtype left_mouth_x = landmark[6];
    		const Dtype left_mouth_y = landmark[7];
    		const Dtype right_mouth_x = landmark[8];
    		const Dtype right_mouth_y = landmark[9];
    		src_center.x = (left_eye_x + right_eye_x + left_mouth_x + right_mouth_x) 
				/ 4;
    		src_center.y = (left_eye_y + right_eye_y + left_mouth_y + right_mouth_y) 
				/ 4;
		} else {
    		src_center.x = 0;
    		src_center.y = 0;
    		for (int i = 0; i < center_ind.size(); ++i) {
      			src_center.x += landmark[center_ind[i] * 2];
      			src_center.y += landmark[center_ind[i] * 2 + 1];
    		}
    		src_center.x /= center_ind.size();
    		src_center.y /= center_ind.size();
  		}
  		return src_center;
	}
	template cv::Point_<float> GetAffineImage_GetSrcCenter(
			const vector<float>& landmark, const vector<int> center_ind);
	template cv::Point_<double> GetAffineImage_GetSrcCenter(
			const vector<double>& landmark, const vector<int> center_ind);


	template <typename Dtype>
	Dtype GetAffineImage_GetScale(const cv::Mat& src, cv::Mat& dst,
    		const vector<Dtype>& landmark, const int norm_mode, 
			const float norm_ratio) {
  		const Dtype left_eye_x = landmark[0];
  		const Dtype left_eye_y = landmark[1];
  		const Dtype right_eye_x = landmark[2];
  		const Dtype right_eye_y = landmark[3];
  		const Dtype left_mouth_x = landmark[6];
  		const Dtype left_mouth_y = landmark[7];
  		const Dtype right_mouth_x = landmark[8];
  		const Dtype right_mouth_y = landmark[9];
  		const Dtype norm_standard_len = MAX(dst.rows, dst.cols) * norm_ratio;
  		Dtype actual_len = norm_standard_len;
  		switch (norm_mode) {
    	case 0: {
			const Dtype deltaX1 = left_eye_x - left_mouth_x;
     	 	const Dtype deltaY1 = left_eye_y - left_mouth_y;
      		const Dtype deltaX2 = right_eye_x - right_mouth_x;
      		const Dtype deltaY2 = right_eye_y - right_mouth_y;
      		actual_len = std::sqrt(deltaX1 * deltaX1 + deltaY1 * deltaY1)
          		+ std::sqrt(deltaX2 * deltaX2 + deltaY2 * deltaY2);
      		actual_len /= Dtype(2.);
      		break;
    		}
    	case 1: {
      		const Dtype left_top_x = MIN(MIN(MIN(left_eye_x, right_eye_x), 
				left_mouth_x), right_mouth_x);
      		const Dtype right_bottom_x = MAX(MAX(MAX(left_eye_x, right_eye_x),
                left_mouth_x), right_mouth_x);
      		const Dtype left_top_y = MIN(MIN(MIN(left_eye_y, right_eye_y),
                left_mouth_y), right_mouth_y);
      		const Dtype right_bottom_y = MAX(MAX(MAX(left_eye_y, right_eye_y),
                left_mouth_y), right_mouth_y);
      		const Dtype deltaX = right_bottom_x - left_top_x;
      		const Dtype deltaY = right_bottom_y - left_top_y;
      		actual_len = std::sqrt((deltaX * deltaX + deltaY * deltaY) / Dtype(2.));
      		break;
    		}
    	case 2: {
      		const Dtype deltaX = left_eye_x - right_eye_x;
      		const Dtype deltaY = left_eye_y - right_eye_y;
      		actual_len = std::sqrt(deltaX * deltaX + deltaY * deltaY);
      		break;
    		}
    	default:
      		cout << "ERROR: Unknow Norm Mode" << endl;
  		}
  		const Dtype scale = actual_len / norm_standard_len;
  		return scale;
	}
	template float GetAffineImage_GetScale(const cv::Mat& src, cv::Mat& dst,
    		const vector<float>& landmark, const int norm_mode,
			const float norm_ratio);
	template double GetAffineImage_GetScale(const cv::Mat& src, cv::Mat& dst,
    		const vector<double>& landmark, const int norm_mode,
			const float norm_ratio);

	template <typename Dtype>
	Dtype GetAffineImage_GetAngle(const vector<Dtype>& landmark) {
  		const Dtype left_eye_x = landmark[0];
  		const Dtype left_eye_y = landmark[1];
  		const Dtype right_eye_x = landmark[2];
  		const Dtype right_eye_y = landmark[3];
  		return atan2((right_eye_y - left_eye_y), (right_eye_x - left_eye_x));
	}
	template float GetAffineImage_GetAngle(const vector<float>& landmark);
	template double GetAffineImage_GetAngle(const vector<double>& landmark);

	template <typename Dtype>
	cv::Mat_<Dtype> Get_Affine_matrix(const cv::Point_<Dtype>& srcCenter, 
			const cv::Point_<Dtype>& dstCenter, const Dtype alpha, 
			const Dtype scale) {
		cv::Mat_<Dtype> M(2, 3);
  		M(0, 0) = scale * cos(alpha);
  		M(0, 1) = scale * sin(alpha);
  		M(1, 0) = -M(0, 1);
  		M(1, 1) = M(0, 0);
  		// 很容易得到M(0, 2)跟M(1, 2)
  		// 方法是：使得dstCenter经过M变换之后是srcCenter
  		M(0, 2) = srcCenter.x - M(0, 0) * dstCenter.x - M(0, 1) * dstCenter.y;
  		M(1, 2) = srcCenter.y - M(1, 0) * dstCenter.x - M(1, 1) * dstCenter.y;
  		return M;
	}

	template cv::Mat_<float> Get_Affine_matrix(const cv::Point_<float>& srcCenter,
    	const cv::Point_<float>& dstCenter, const float alpha, const float scale);
	template cv::Mat_<double> Get_Affine_matrix(const cv::Point_<double>& srcCenter,
    	const cv::Point_<double>& dstCenter, const double alpha, const double scale);

	template <typename Dtype>
	void mAffineWarp(const cv::Mat_<Dtype>& M, const cv::Mat& srcImg, cv::Mat& dstImg,
    		const bool fill_type, const uchar value) {
  		if (dstImg.empty()) 
				dstImg = cv::Mat(srcImg.size(), srcImg.type(), cv::Scalar::all(0));
  		for (int y = 0; y < dstImg.rows; ++y) {
    		for (int x = 0; x < dstImg.cols; ++x) {
      			Dtype fx = M(0, 0) * x + M(0, 1) * y + M(0, 2);
      			Dtype fy = M(1, 0) * x + M(1, 1) * y + M(1, 2);
      			int sy = cvFloor(fy);
      			int sx = cvFloor(fx);
      			if (fill_type && (sy < 1 || sy > srcImg.rows - 2 || sx < 1 
					|| sx > srcImg.cols - 2)) {
        			for (int k = 0; k < srcImg.channels(); ++k) {
         	 			dstImg.at<cv::Vec3b>(y, x)[k] = value;
        			}
        			continue;
      			}
      			fx -= sx;
      			fy -= sy;
      			sy = MAX(1, MIN(sy, srcImg.rows - 2));
      			sx = MAX(1, MIN(sx, srcImg.cols - 2));
      			Dtype w_y0 = std::abs(1.0f - fy);
      			Dtype w_y1 = std::abs(fy);
      			Dtype w_x0 = std::abs(1.0f - fx);
      			Dtype w_x1 = std::abs(fx);
      			for (int k = 0; k < srcImg.channels(); ++k) {
        			dstImg.at<cv::Vec3b>(y, x)[k] = (srcImg.at<cv::Vec3b>(sy, sx)[k] 
						* w_x0 * w_y0 + srcImg.at<cv::Vec3b>(sy + 1, sx)[k] * w_x0 
						* w_y1 + srcImg.at<cv::Vec3b>(sy, sx + 1)[k] * w_x1 * w_y0
            			+ srcImg.at<cv::Vec3b>(sy + 1, sx + 1)[k] * w_x1 * w_y1);
      			}
    		}
  		}
	}
	template void mAffineWarp<float>(const cv::Mat_<float>& M, const cv::Mat& srcImg, 
			cv::Mat& dstImg, const bool fill_type, const uchar value);
	template void mAffineWarp<double>(const cv::Mat_<double>& M, const cv::Mat& srcImg, 
			cv::Mat& dstImg, const bool fill_type, const uchar value);

} // namespace jaffe
