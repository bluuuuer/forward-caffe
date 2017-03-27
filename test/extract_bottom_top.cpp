#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "net.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using jaffe::JBlob;
using jaffe::JNet;

static const int BLOB_MAX_CHANNEL = 100;

cv::Mat target_channels(const shared_ptr<JBlob<float> >& target_blob, const int i) {
	cv::Mat imgs[BLOB_MAX_CHANNEL + 1];

	const float *rawdata;
	const int w = target_blob->GetWidth();
	const int h = target_blob->GetHeight();
	const int ch = target_blob->GetChannels();
	
	int nrow = std::ceil(std::sqrt(ch));
	int ncol = nrow;

	imgs[ch] = cv::Mat::zeros(nrow * h, ncol * w, CV_8U);
	for (int j = 0; j < ch; j ++) {
		rawdata = target_blob->GetData() + target_blob->GetOffset(i, j);
		imgs[i] = cv::Mat(cv::Size(w, h), CV_32FC1, (void *) rawdata, 
			cv::Mat::AUTO_STEP);
		cv::normalize(imgs[j], imgs[j], 255, 0, cv::NORM_MINMAX);
		imgs[j].convertTo(imgs[j], CV_8U);
	}
	cv::merge(imgs, 3, imgs[3]);

	return imgs[3];
}

int main(int argc, char* argv[]) {

	string trained_file = argv[1];
	string net_proto = argv[2];
	string blob_name = argv[3];
	string output_dir = argv[4];

	shared_ptr<JNet<float> > net(new JNet<float>(net_proto, jaffe::TEST));
	net->CopyTrainedLayersFrom(trained_file);

	const shared_ptr<JBlob<float> > target_blob = net->GetBlobByName(blob_name);
	char keystr[1000];

	net->NetForward();
	const int batch = target_blob->GetNum();
	for (int i = 0; i < batch; i ++) {
		cv::Mat output;
		output = target_channels(target_blob, i);
		snprintf(keystr, 100, "%s/%s-%d.png", output_dir.c_str(), blob_name.c_str(),
			i);
		cv::imwrite(keystr, output);
	}
}
