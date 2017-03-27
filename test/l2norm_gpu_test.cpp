#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/time.h>
#include <dirent.h>
#include <math.h>

#include "net.h"

using namespace std;
using namespace jaffe;

static bool MyLessThan(const pair<pair<string, string>, float>& l,
				const pair<pair<string, string>, float>& r) {
	return l.second < r.second;
}

float CalDis(const string file_1, const string file_2, const int num) {
	ifstream fs_1(file_1);
	ifstream fs_2(file_2);
	string line_1, line_2;
	int index = 0;
	float sum = 0.0;
	if (fs_1 && fs_2) {
		while(getline(fs_1, line_1) && getline(fs_2, line_2)) {
			float x_1 = atof(line_1.c_str());
			float x_2 = atof(line_2.c_str());
			sum += (x_1 - x_2) * (x_1 - x_2);
			index ++;
		}
	}
	fs_1.close();
	fs_2.close();
	if (index != num) {
		return -1.0;
	}
	return sqrt(sum);
}

int main(int argc, char** argv) {
	timeval t_start_all, t_end_all;
	double timeuse_all;
	gettimeofday(&t_start_all, NULL);

	timeval t_start, t_end;
	double timeuse;

	// 各种参数读入
	string trained_file = argv[1];
	string deploy_file = argv[2];
	string image_folder = argv[3];
	string landmark_folder = argv[4];
	string affine_image_folder = argv[5];
	string output_folder = argv[6];

 	int gpu_id = 1;
	Jaffe::SetDevice(gpu_id);
	cout << "Use GPU with device ID: " << gpu_id << endl;
	//cudaDeviceProp device_prop;
	//cudaGetDeviceProperties(&device_prop, gpu_id);
	//cout << "GPU device name: " << device_prop.name << endl << endl;
	Jaffe::SetMode(Jaffe::GPU);	

	// 网络初始化，reshape
	JNet<float>* net = new JNet<float>(deploy_file, TEST);
	cout << "Done Net()" << endl << endl;
	net->CopyTrainedLayersFrom(trained_file);
	cout << "Done CopyTrainedLayersFrom()" << endl << endl;
	net->Reshape();
	cout << endl;

	// 读取图片和标记存储路径
	int img_count = 0;
	vector<pair<string, string> > path_list;
	vector<string> image_name;
	vector<string> output_file;
	DIR *dirptr = NULL;
	struct dirent *entry;
	if (!(dirptr = opendir(image_folder.data()))) {
		cout << "Error opendir " << image_folder << "!!!" << endl;
	}
	cout << "Image Files && Landmark Files" << endl;
	while (entry = readdir(dirptr)) {
		if (strncmp(entry->d_name, ".", 1) == 0)
			continue;
		string dname(entry->d_name);
		image_name.push_back(dname.substr(0, dname.find_last_of('.')));
		output_file.push_back(output_folder + image_name[img_count] + ".txt");
		string landmark_path = landmark_folder + image_name[img_count] + ".txt";
		string image_path = image_folder + dname;
		path_list.push_back(make_pair(image_path, landmark_path));
		cout << path_list[img_count].first << "\t\t" << path_list[img_count].second 
			<< endl;
		img_count ++;
	}
	cout << endl << "image count = " << img_count << endl << endl;
	// 读取图片和标记
	vector<vector<float> > landmark_list;
	vector<cv::Mat> image_list;
	string line;
	for (int i = 0; i < img_count; i++){
	 	vector<float> temp;
		//cout << "imread(" << path_list[i].first << ")" << endl;
		cv::Mat image = cv::imread(path_list[i].first);
		image_list.push_back(image);
		ifstream lm_in(path_list[i].second);
		if (!lm_in) {
			cout << "ERROR: No such landmark file " << path_list[i].second << endl;
		} else {
			while (getline(lm_in, line)) {
				temp.push_back(atof(line.c_str()));
			}
			if (temp.size() != 10) {
				cout << "ERROR: Not enough(" << landmark_list[i].size() << 
					") parameter for landmark" << endl;
			}
			landmark_list.push_back(temp);
		}
		lm_in.close();
	}

	// GetAffineImage 并保存
	vector<int> center_ind;
	center_ind.push_back(-1);
	cout << "Affine Files:" << endl;
	for (int i = 0; i < img_count; i ++) {
		JBlob<float>* input_blob = net->GetInputBlobs()[0];

		cv::Mat affine_img(input_blob->GetHeight(), input_blob->GetWidth(), CV_8UC3);
		GetAffineImage(image_list[i], affine_img, landmark_list[i], center_ind, 1, 0.3, 
			false, 0);
		string affine_file = affine_image_folder + image_name[i] + ".jpg";
		cout << "\t" << affine_file << endl;
		cv::imwrite(affine_file, affine_img);
		// 把图片数据输入到网络中
		ImageDataToBlob(input_blob, i, affine_img);
		cout << "Done input " << i << "th image to net" << endl;
	}
	cout << endl;

	cout << "NetForward: " << endl;
	int test_times = 1;
	for (int i = 0; i < test_times; i ++) { 
		cout << "test[" << i << "]: " << endl;
		gettimeofday(&t_start, NULL);
		net->NetForward();
//		cout << "Done NetForward()" << endl;
		gettimeofday(&t_end, NULL);
		timeuse = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)
			/ 1000000.0;
		cout << endl << "Net Forward Time Use: " << timeuse << " s" << endl << endl;
	}

		// 从网络中获取输出并写入指定文件夹
	vector<JBlob<float>*> last_layer_bottom = net->GetTopBlobsByIndex(-2); 
	cout << "GetNum() = " << last_layer_bottom[0]->GetNum() << endl;
	cout << "GetChannels() = " << last_layer_bottom[0]->GetChannels() << endl;
	
	ofstream out;
	cout << "Write Output Files..." << endl;
	const float* output = last_layer_bottom[0]->GetData();
	for (int i = 0; i < last_layer_bottom[0]->GetNum(); i++) {
		out.open(output_file[i], ios::trunc | ios::out);
		if (!out) {
			cout << "Error open file " << output_file[i] << endl;
		}
		for (int j = 0; j < last_layer_bottom[0]->GetChannels(); j++) {
			int index = i * last_layer_bottom[0]->GetChannels() + j;
			//cout << "last_layer_bottom[0] " << index << ") : " << output[index]	
				//<< endl;
			out << output[index] << endl;
		}
		out.close();
	}
	cout << "Done" << endl << endl;

	gettimeofday(&t_end_all, NULL);
	timeuse_all = t_end_all.tv_sec - t_start_all.tv_sec + (t_end_all.tv_usec - 
		t_start_all.tv_usec) / 1000000.0;
	cout << "Time Use: " << timeuse_all << " s" << endl << endl;

	vector<pair<pair<string, string>, float> > result;
	for (int i = 0; i < img_count - 1; i ++) {
		for (int j = i + 1; j < img_count; j ++) {
			result.push_back(make_pair(make_pair(image_name[i], image_name[j]), 
				CalDis(output_file[i], output_file[j], 160)));
		}
	}
	
	int pair_count = img_count * (img_count - 1) / 2;
	partial_sort(result.begin(), result.begin() + pair_count, result.end(), 
		MyLessThan);

	cout << "Compare Result: " << endl;
	for (int i = 0; i < pair_count; i ++) {
		cout << "[" << result[i].first.first << ", " << result[i].first.second 
			<< "]: " << result[i].second << endl;
	}

	return 0;
}
