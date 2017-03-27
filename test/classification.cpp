#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <sys/time.h>
#include <iomanip>

#include "net.h"

using namespace Eigen;

using namespace jaffe;  // NOLINT(build/namespaces)
using std::string;
using std::cout;
using std::endl;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file, const string& trained_file, 
			 const string& mean_file, const string& label_file, const int gpu_id);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<JNet<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
}; // class Classifier

Classifier::Classifier(const string& model_file, const string& trained_file,
                       const string& mean_file, const string& label_file,
					   const int gpu_id) {

  /* Load the network. */
	Jaffe::SetMode(Jaffe::GPU);
	Jaffe::SetDevice(gpu_id);
	//Jaffe::SetMode(Jaffe::CPU);
  net_.reset(new JNet<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);


  JBlob<float>* input_layer = net_->GetInputBlobs()[0];
  num_channels_ = input_layer->GetChannels();
  input_geometry_ = cv::Size(input_layer->GetWidth(), input_layer->GetHeight());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  //JBlob<float>* output_layer = net_->GetOutputBlobs()[0];
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	//cout << "mean_file: " << mean_file << endl;
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  JBlob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  /*================test=============
  	int mean_blob_num = mean_blob.GetCount();
	cout << "mean_blob_num = " << mean_blob_num << endl;
	for (int i = 0; i < mean_blob_num; i++) {
		string type = typeid(mean_blob.GetData()[i]) == typeid(float) ? "float"
			: "not float";
		cout << "mean_blob " << i << ") = " << mean_blob.GetData()[i] << 
			"\t type is " << type << endl;
	}*/

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.GetMutableData();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.GetHeight(), mean_blob.GetWidth(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.GetHeight() * mean_blob.GetWidth();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  		
  /*================test=============
  	int row = mean_.rows;
	int col = mean_.cols * mean_.channels();
	cout << "mean_.size = " << row << " * " << col << endl;
	for (int i = 0; i < row; i++) {
		float* data = mean_.ptr<float>(i);
		for (int j = 0; j < col; j++){
			cout << data[j] << "\t";
		}
		cout << endl;
	}*/

} // SetMean()

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  JBlob<float>* input_layer = net_->GetInputBlobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);
  
  /*================test=============
  	int input_num = input_layer->GetCount();
	for (int i = 0; i < input_num; i++) {
		string type = typeid(input_layer->GetData()[i]) == typeid(float) ? "float"
			: "not float";
		cout << "input_layer " << i << ") = " << input_layer->GetData()[i] << 
			"\t type is " << type << endl;
	}*/

  net_->NetForward();

  /* Copy the output layer to a std::vector */
  JBlob<float>* output_layer = net_->GetOutputBlobs()[0];
  const float* begin = output_layer->GetData();
  const float* end = begin + output_layer->GetChannels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  JBlob<float>* input_layer = net_->GetInputBlobs()[0];

  int width = input_layer->GetWidth();
  int height = input_layer->GetHeight();
  float* input_data = input_layer->GetMutableData();
  for (int i = 0; i < input_layer->GetChannels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  //reinterpret_cast<float*>(input_channels->at(0).data)
        //== net_->GetInputBlobs()[0]->GetData();
}

int main(int argc, char** argv) {
	//std::cout << test() << std::endl;
	timeval t1, t2;
	double timeuse;
	gettimeofday(&t1, NULL);

  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }


  string model_file = argv[1];
  string trained_file = argv[2];
  string mean_file = argv[3];
  string label_file = argv[4];
  const int gpu_id = atoi(argv[5]);
  cout << "Using GPU " << gpu_id << endl;
  Classifier classifier(model_file, trained_file, mean_file, label_file, gpu_id);

  string file = argv[6];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }

	gettimeofday(&t2, NULL);
	timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	std::cout << "Use Time: " << timeuse << std::endl;
}
