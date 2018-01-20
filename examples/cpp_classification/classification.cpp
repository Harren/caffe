#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <time.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces) //�����ռ�caffe
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {	//C++��Classifier�����Ķ�����ʵ�ֶ�����һ��.cpp�ļ��У���main������ʹ�õ���������
 public:
  Classifier(const string& model_file,		//ģ�������ļ�deploy.prototxt
             const string& trained_file,	//ѵ���õ�ģ��.caffemodel
             const string& mean_file,		//��ֵ�ļ�mean.binaryproto
             const string& label_file);		//��ǩ�ı�labels.txt---������������������

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 2);	//���ຯ��������Matͼ�����ݺ�top N��Ĭ��Ϊtop 5�������ĳ�top 2

  ////////////////////����
  std::vector<Prediction> Classify(const cv::Mat& img, int N, int task_index);	//���ط��ຯ��������Matͼ�����ݡ�top N������������
  ///////////////////////															//��������Ϊ0ʱ����ǰ�򴫲�����������0��Ԥ����������Ϊ0������Ҫ�ٽ���ǰ�򴫲��ˣ�ֱ�ӷ��ظ�����������Ԥ������

 private:
  void SetMean(const string& mean_file);	//����mean_file��ֵ�ļ����ɾ�ֵMatͼ��mean_

  std::vector<float> Predict(const cv::Mat& img);

  ////////////////////����
  std::vector<float> Predict(const cv::Mat& img, int task_index);	//����Ԥ�⺯������������Ϊ0ʱ����ǰ�򴫲�����������0��Ԥ����������Ϊ0������Ҫ�ٽ���ǰ�򴫲��ˣ�ֱ�ӷ��ظ�����������Ԥ������
  ///////////////////////

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;	//����net_��caffe�г�Ա�������к�׺_�����ﱣ�����ַ���
  cv::Size input_geometry_;		//����ά��(width,height)
  int num_channels_;			//ͨ����
  cv::Mat mean_;				//��ֵMatͼ��
  std::vector<string> labels_;	//��ǩ��������������

  std::vector<vector<string> > all_task_labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {	//���صĹ��캯��
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);	//CPUģʽ
#else
  Caffe::set_mode(Caffe::GPU);	//GPUģʽ
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));	//����nte_��������ģ�ͽṹ��TESTģʽ
  net_->CopyTrainedLayersFrom(trained_file);	//����net_����ѵ���õ�����ģ��

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";		//����������Blob��Ŀ��ֻ����1������net_->input_blobs()[0]
//  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";	//����������Blob��Ŀ	////////////////////////////////////////////////////////////////////////////////////////////////

  Blob<float>* input_layer = net_->input_blobs()[0];	//����������Blob�������׸�Blobָ�룬Blob����ά����(width_,height_,channels_,num_)
  num_channels_ = input_layer->channels();				//��ȡ����Blob���׸�����ͨ����
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());	//��ȡ����Blob���׸����ļ���ά�ȣ�cv::Size

  /* Load the binaryproto mean file. */
  SetMean(mean_file);	//����mean_file��ֵ�ļ������ɾ�ֵͼ��mean_

  /* Load labels. */
  std::ifstream labels(label_file.c_str());		//�����ļ���labels��c_str()��������һ��ָ������C�ַ�����ָ��, �����뱾string������ǩ�ı�·��label_file����ͬ
  CHECK(labels) << "Unable to open labels file " << label_file;
  /*
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
	*/
  //��ȡ��������ǩ���ձ�
  int task_num;
  labels >> task_num;
  vector<int> label_num(task_num);
  int i, j, index;
  for (i = 0; i < task_num; i++)
	  labels >> label_num[i];
  for (i = 0; i < task_num; i++){
	  vector<string> my_labels(label_num[i]);
	  for (j = 0; j < label_num[i]; j++)
		  labels >> index >> my_labels[j];
	  all_task_labels_.push_back(my_labels);
  }

  /*
  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())	//��ǩ�ı��е�������Ҫ������Blob��ͨ��������ʵ���Ƿֵ�����������һ��	////////////////////////////////////////////////////////////////////////////////////////////////
    << "Number of labels is different from the output layer dimension.";
	*/
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {	//����top N ������������������N=1
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {	//Classifier����Ա���������ڷ��࣬����top N ��Ԥ������
  std::vector<float> output = Predict(img);		//��ȡԤ���ĸ���float����

  N = std::min<int>(labels_.size(), N);			//��N�ȱ�ǩ����������������ȡNΪ��ǩ��������
  std::vector<int> maxN = Argmax(output, N);	//����top N ������
  std::vector<Prediction> predictions;			//Ԥ��������������top N ���������͸���ֵ��
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));	//���ص�Ԥ������Ϊ��top N ���������͸���ֵ
  }

  return predictions;
}

//////////////////////���������غ���Classify
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N, int task_index)
{
	std::vector<float> output = Predict(img, task_index);		//��ȡԤ���ĸ���float����

	N = std::min<int>(all_task_labels_[task_index].size(), N);			//��N�ȱ�ǩ����������������ȡNΪ��ǩ��������
    std::vector<int> maxN = Argmax(output, N);	//����top N ������
	std::vector<Prediction> predictions;			//Ԥ��������������top N ���������͸���ֵ��


    for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(all_task_labels_[task_index][idx], output[idx]));	//���ص�Ԥ������Ϊ��top N ���������͸���ֵ
	}
	return predictions;
}
////////////////////////////////

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);	//��mean.binaryproto�ж�ȡ��blob_proto

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);	//��BlobProto��Blob
  CHECK_EQ(mean_blob.channels(), num_channels_)	//��ֵ�ļ�ͼ��ͨ����Ҫ������ͼ��ͨ����һ��
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();	//��д����cpu data��32-bit float BGR or grayscale
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);	//���캯������Mat
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();	//Ϊ�˷���ͨ��
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);	//ͨ���ں�,�õ�Matͼ������mean

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean); //����ÿ��ͨ���ľ�ֵ������һ����ά����������ImageNet���ݼ���Ϊ104,117,123��R,G,B��
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);	//����ά������ʼ����ͨ������һ��Matͼ��mean_
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);	//������Blob������ת���ɸ�ͨ����cv::Matͼ����������

  Preprocess(img, &input_channels);	//����ɫ�ռ�ת�������š�ȥ��ֵ��Ԥ����������ͨ���������ĸ�ͨ��Mat

  net_->Forward();	//ǰ�򴫲�

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];	//�õ�ǰ�򴫲���������Blob
  const float* begin = output_layer->cpu_data();		//����Blob��������ָ��
  const float* end = begin + output_layer->channels();	//����Blob������δָ��
  return std::vector<float>(begin, end);				//����Ԥ�����ʽ�������
}

/////////////////����������Predict
std::vector<float> Classifier::Predict(const cv::Mat& img,int task_index) {
	if (task_index == 0){
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);	//������Blob������ת���ɸ�ͨ����cv::Matͼ����������

		Preprocess(img, &input_channels);	//����ɫ�ռ�ת�������š�ȥ��ֵ��Ԥ����������ͨ���������ĸ�ͨ��Mat

		net_->Forward();	//ǰ�򴫲�
	}
	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[task_index];	//�õ�ǰ�򴫲���������Blob
	const float* begin = output_layer->cpu_data();		//����Blob��������ָ��
	const float* end = begin + output_layer->channels();	//����Blob������δָ��
	return std::vector<float>(begin, end);				//����Ԥ�����ʽ�������
}
//////////////////////

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {	//������Blob������ת����cv::Matͼ������
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {		//����ɫ�ռ�ת�������š�ȥ��ֵ��Ԥ����������ͨ���������ĸ�ͨ��Mat
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)	//��ɫת��
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
  if (sample.size() != input_geometry_)	//����
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);	//ȥ��ֵ

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);		//ͨ������

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];	//ģ�������ļ�deploy.prototxt
  string trained_file = argv[2];	//ѵ���õ�����ģ��network.caffemodel
  string mean_file    = argv[3];	//��ֵ�ļ�mean.binaryproto
  string label_file   = argv[4];	//��ǩ�ļ�labels.txt---����������������Ӧ��
  Classifier classifier(model_file, trained_file, mean_file, label_file);	//����Classifier�����󣬹��캯�������������������ĸ�

  string file = argv[5];	//���������Ե�ͼ��img.jpg

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  /*
  std::vector<Prediction> predictions = classifier.Classify(img);	//����Classifier�������ķ��ຯ��Classify����������Ϊcv::Matͼ�����ݣ�����

  /* Print the top N predictions. */
  /*
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
  */

  //clock_t time_start, time_end;
  //time_start = clock();
  int TASK_NUM = 6;	//��������
  int TOP_N = 1;	//top N
  for (int index = 0; index < TASK_NUM; index++)
  {
	  std::vector<Prediction> predictions = classifier.Classify(img, TOP_N, index);	//���ж��������࣬��������������(����Ϊindex=0ʱ����ǰ�򴫲�����������0��Ԥ������������Ϊindex��=0ʱ������Ҫ����ǰ�򴫲���ֱ�ӷ�������index��Ԥ������)

      for (size_t i = 0; i < predictions.size(); ++i) {	//��ӡԤ����top N
          Prediction p = predictions[i];
		  std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			  << p.first << "\"" << std::endl;
	  }
  }
  //time_end = clock();
  //double duration = (double)(time_end - time_start) / CLOCKS_PER_SEC;
  //std::cout << "duaration:" << duration << std::endl;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
