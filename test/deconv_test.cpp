#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

#include "net.h"

using std::string;
using std::cout;
using std::endl;
using namespace jaffe;

int main(int argc, char** argv) {

	string trained_file = argv[1];
	string deploy_file = argv[2];
	string image_file = argv[3];
	string output_file = argv[4];

	cv::Mat img = cv::imread(image_file);
	/*cout << "=================img=======================" << endl;
	for (int h = 0; h < img.rows; h++) {
		uchar* data = img.ptr<uchar>(h);
		for (int w = 0; w < img.cols; w++) {
			cout << (float)data[w] << "\t";
		}
		cout << endl;
	}*/
	//cout << "img.rows = " << img.rows << endl;
	//cout << "img.cols = " << img.cols << endl;
	cv::Size img_size(27, 35);
	cv::Mat img_resized;
	cv::resize(img, img_resized, img_size); 
	//cout << "img_resized.rows = " << img_resized.rows << endl;
	//cout << "img_resized.cols = " << img_resized.cols << endl;
	cv::Mat img_one;
	cv::cvtColor(img_resized, img_one, cv::COLOR_BGR2GRAY);
	//cout << "img_one.rows = " << img_one.rows << endl;
	//cout << "img_one.cols = " << img_one.cols << endl;

	JNet<float>* net = new JNet<float>(deploy_file, TEST);
	net->CopyTrainedLayersFrom(trained_file);
	net->Reshape();
	//cout << "Reshape() Done" << endl;
	
	JBlob<float>* input_blob = net->GetInputBlobs()[0];
	ImageDataToBlob(input_blob, 0, img_one);
	for (int i = 0; i < input_blob->GetCount(); i++) {
		input_blob->GetMutableData()[i] /= 1024;
	}
	/*
	cout << "=================input_blob=======================" << endl;
	for (int h = 0; h < img_one.rows; h++) {
		for (int w = 0; w < img_one.cols; w++) {
			int i = h * img_one.cols + w;
			cout << input_blob->GetData()[i] << "\t";
		}
		cout << endl;
	}
	*/

	net->NetForward();

	/*
	for (int i = 0; i < net->GetLayers().size(); i++) {
		vector<JBlob<float>*> temp_layer_top = net->GetTopBlobsByIndex(i);
		for (int j = 0; j < temp_layer_top.size(); j++) {
			cout << "=======top blob " << i << "[" << j << "] =======" << endl;
			cv::Mat temp_img(temp_layer_top[j]->GetHeight(), 
				temp_layer_top[j]->GetWidth(), CV_8UC1);
			BlobToGreyImage(temp_layer_top[j], (float)1024., &temp_img);
			cout << temp_img << endl << endl;
		}
	}
	*/

	vector<JBlob<float>*> last_layer_top = net->GetTopBlobsByIndex(-1); 
	cv::Mat output_img(last_layer_top[0]->GetHeight(), last_layer_top[0]->GetWidth(),
		CV_8UC1);
	BlobToGreyImage(last_layer_top[0], (float)1024., &output_img);
	/*cout << "=================output_blob=======================" << endl;
	for (int h = 0; h < output_img.rows; h++) {
		for (int w = 0; w < output_img.cols; w++) {
			int index = h * output_img.cols + w;
			cout << last_layer_top[0]->GetData()[index] << "\t";
		}
		cout << endl;
	}*/

	cv::imwrite(output_file, output_img);
	//cout << "=================output_img=======================" << endl;
	//cout << output_img << endl << endl;

	return 0;
}
