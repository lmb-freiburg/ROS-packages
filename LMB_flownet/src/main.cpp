/** 
 * Eddy Ilg, 2018
 * Nikolaus Mayer, 2019
 *
 * This module provides optical flow estimation with FlowNet.
 *
 * Input topic: /uvc_cam1_rect_mono
 * Output topic: /flownet
 *
 * Parameter flownet_variant:
 *
 * Name of the flownet variant. Following options are available
 * (ordered from low quality/fast to high quality/slow):
 *
 * FlowNet2-s
 * FlowNet2-ss
 * FlowNet2-css-ft-sd
 * FlowNet2-cssR (same as above, but including highres refinement)
 * FlowNet2-CSS-ft-sd
 * FlowNet2 (complete FlowNet2, best but slowest)
 *
 */

/// System/STL
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <time.h>

/// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// ROS
#include <ros/ros.h>
#include <ros/package.h>  // finds package paths
#include <sensor_msgs/Image.h>
#include <sensor_msgs/fill_image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

/// Caffe
#include "caffe/caffe.hpp"

using namespace caffe;


std::string mode{"adjacent-frames"};
bool do_forward_backward_consistency{false};
float FBC_pixel_difference_threshold{2.0f};
bool do_match_feasibility_check{false};


class NetWrapper {
  public:
    /**
     * @brief Constructor
     * @param model_file Prototxt file describing the graph structure of the
     *                   neural network
     * @param weights_file Protobuf or H5 file containing the trained network
     *                     parameter sets (the "weights")
     */
    NetWrapper( const std::string& model_file, const std::string& weights_file);

    ~NetWrapper();

    /**
     * @brief Easier access to network inputs
     * @param input_channels Empty; will get one element for each input channel
     * @param index Index of the input_layer to map: 0 or 1
     */
    void WrapInputLayer( std::vector<cv::Mat>* input_channels, int index);

    /**
     * @brief Estimate pixelwise disparities for a stereo pair
     * @param img_0 Image at time t
     * @param img_1 Image at time t+1
     * @param output Output paramieter; flow (displacements for each pixel location from time t to time t+1)
     */
    void Predict( const cv::Mat& img_0, const cv::Mat& img_1, cv::Mat& output);

    /// The neural network
    Net<float>* net_;
    /// Size of the network's inputs and outputs
    cv::Size input_geometry_;
};

NetWrapper::NetWrapper(const string& model_file, const string& weights_file)
{
  /// We must use GPU mode as the DispNet does not have a working
  /// CPU-only implementation
  Caffe::set_mode(Caffe::GPU);

  /// Create network
  net_ = new Net<float>(model_file, TEST);

  /// Initialize network from pretrained weights
  if (weights_file != "")
    net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs.";

  input_geometry_ = cv::Size(net_->input_blobs()[0]->width(),
                             net_->input_blobs()[0]->height());
}

NetWrapper::~NetWrapper()
{
  delete net_;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void NetWrapper::WrapInputLayer(std::vector<cv::Mat>* input_channels, int index) 
{
  Blob<float>* input_layer = net_->input_blobs()[index];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void NetWrapper::Predict(const cv::Mat& img_0,
                         const cv::Mat& img_1,
                         cv::Mat& output)
{
  /// Feed data
  std::vector<cv::Mat> input_channels_0;
  std::vector<cv::Mat> input_channels_1;
  WrapInputLayer(&input_channels_0, 0);
  WrapInputLayer(&input_channels_1, 1);

  for (int y = 0; y < input_channels_0[0].rows; ++y) {
    for (int x = 0; x < input_channels_0[0].cols; ++x) {
      input_channels_0[0].at<float>(y,x) = img_0.at<cv::Vec3f>(y,x)[0];
      input_channels_0[1].at<float>(y,x) = img_0.at<cv::Vec3f>(y,x)[1];
      input_channels_0[2].at<float>(y,x) = img_0.at<cv::Vec3f>(y,x)[2];
    }
  }
  CHECK(reinterpret_cast<float*>(input_channels_0[0].data) ==
      net_->input_blobs()[0]->cpu_data());\

    for (int y = 0; y < input_channels_1[0].rows; ++y) {
      for (int x = 0; x < input_channels_1[0].cols; ++x) {
        input_channels_1[0].at<float>(y,x) = img_1.at<cv::Vec3f>(y,x)[0];
        input_channels_1[1].at<float>(y,x) = img_1.at<cv::Vec3f>(y,x)[1];
        input_channels_1[2].at<float>(y,x) = img_1.at<cv::Vec3f>(y,x)[2];
      }
    }
  CHECK(reinterpret_cast<float*>(input_channels_1[0].data) == net_->input_blobs()[1]->cpu_data());

  /// Network forward pass
  net_->Forward();

  /// Extract output
  int height = input_geometry_.height;
  int width = input_geometry_.width;

  Blob<float>* output_layer = net_->output_blobs()[net_->num_outputs()-1];

  cv::Mat prediction(height*2, width,
      CV_32FC1, output_layer->mutable_cpu_data());

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      output.at<cv::Vec2f>(y,x)[0] = prediction.at<float>(y,x);
      output.at<cv::Vec2f>(y,x)[1] = prediction.at<float>(y+height,x);
    }
  }
}

NetWrapper* classifier_ptr;
ros::Publisher flow_pub_ptr;

/// These variables are used to store and reference the last image
uint32_t last_img_msg_seq;
cv::Mat last_img;

/// Status information 
time_t last_fps_time=0; // ros::Time somehow doesn't work 
int processed_frames=0;
int skipped_frames=0;

/**
 * @brief Callback for new image. This method
 *        feeds the image data to the neural network retrieves the
 *        result, and publishes a disparity map.
 *        NOTE: ROS messages are not guaranteed to be in sequence.
 *              The routine checks if the image received is a valid
 *              sucessor of the previous image, in case it is not,
 *              no flow is computed!
 * @param img_msg Image topic message
 */
void inputImage_callback(const sensor_msgs::ImageConstPtr& img_msg)
{
  int seq_diff = img_msg->header.seq - last_img_msg_seq;
  last_img_msg_seq = img_msg->header.seq;

  /// Copy data from image 0 message. This can be done way easier,
  /// but cv::Mat::clone is broken on my dev machine (version conflicts?)
  cv_bridge::CvImagePtr cv_ptr =
    cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  cv::Mat img_1(cv_ptr->image.rows, cv_ptr->image.cols, CV_32FC3);
  for (int y = 0; y < cv_ptr->image.rows; ++y) {
    for (int x = 0; x < cv_ptr->image.cols; ++x) {
      img_1.at<cv::Vec3f>(y,x)[0] = cv_ptr->image.at<cv::Vec3b>(y,x)[0];
      img_1.at<cv::Vec3f>(y,x)[1] = cv_ptr->image.at<cv::Vec3b>(y,x)[1];
      img_1.at<cv::Vec3f>(y,x)[2] = cv_ptr->image.at<cv::Vec3b>(y,x)[2];
    }
  }

  /// Check that images are in sequence and last_img was set before
  if(/*seq_diff != 1 ||*/ last_img.cols==0)
  {
    last_img = img_1;
    skipped_frames += seq_diff - 1;
    return;
  }

  /// Display status information
  if(last_fps_time == 0)
    last_fps_time = time(0);

  time_t fps_passed = time(0) - last_fps_time;

  if(fps_passed>5)
  {
    float pfps = float(processed_frames)/float(fps_passed);
    float sfps = float(skipped_frames)/float(fps_passed);

    processed_frames = 0;
    skipped_frames = 0;

    ROS_INFO("Frame rates:  processed = %5.2f/s, skipped = %5.2f/s", pfps, sfps);

    last_fps_time = time(0);
  }        

  /// Previous image
  cv::Mat& img_0 = last_img;

  /// Image -> NeuralNetwork -> Result
  cv::Mat prediction(classifier_ptr->input_geometry_.height,
                     classifier_ptr->input_geometry_.width, CV_32FC2);
  classifier_ptr->Predict(img_0, img_1, prediction);

  if (do_forward_backward_consistency) {
    cv::Mat prediction_r(classifier_ptr->input_geometry_.height,
                         classifier_ptr->input_geometry_.width, CV_32FC2);
    classifier_ptr->Predict(img_1, img_0, prediction_r);
    auto F = [](const cv::Mat& map, float x, float y, size_t c) -> float {
      if (x < 0 or x >= map.cols or
          y < 0 or y >= map.rows)
        return 0.f;
      const int xi = x;
      const int yi = y;
      const int xj = (x < map.cols-1 ? x+1 : x);
      const int yj = (y < map.rows-1 ? y+1 : y);
      const float xf{x-xi};
      const float yf{y-yi};
      return (1-xf)*(1-yf)*map.at<cv::Vec2f>(yi,xi)[c] +
             (  xf)*(1-yf)*map.at<cv::Vec2f>(yi,xj)[c] +
             (1-xf)*(  yf)*map.at<cv::Vec2f>(yj,xi)[c] +
             (  xf)*(  yf)*map.at<cv::Vec2f>(yj,xj)[c];
    };
    for (int y = 0; y < prediction.rows; ++y) {
      for (int x = 0; x < prediction.cols; ++x) {
        const float x1{F(prediction, x, y, 0)};
        const float y1{F(prediction, x, y, 1)};
        const float xres{x1 + F(prediction_r, x+x1, y+y1, 0)};
        const float yres{y1 + F(prediction_r, x+x1, y+y1, 1)};
        const float EPE{std::sqrt(xres*xres + yres*yres)};
        if (EPE >= FBC_pixel_difference_threshold) {
          prediction.at<cv::Vec2f>(y,x)[0] = std::numeric_limits<float>::signaling_NaN();
          prediction.at<cv::Vec2f>(y,x)[1] = std::numeric_limits<float>::signaling_NaN();
        }
      }
    }
  }
  if (do_match_feasibility_check) {
    for (int y = 0; y < prediction.rows; ++y) {
      for (int x = 0; x < prediction.cols; ++x) {
        const float xf{prediction.at<cv::Vec2f>(y,x)[0]};
        const float yf{prediction.at<cv::Vec2f>(y,x)[1]};
        if ((x+xf) < 0 or (x+xf) >= prediction.cols or
            (y+yf) < 0 or (y+yf) >= prediction.rows) {
          prediction.at<cv::Vec2f>(y,x)[0] = std::numeric_limits<float>::signaling_NaN();
          prediction.at<cv::Vec2f>(y,x)[1] = std::numeric_limits<float>::signaling_NaN();
        }
      }
    }
  }

  /// Publish flow field
  sensor_msgs::Image flow_img;
  flow_img.header.stamp = img_msg->header.stamp;

  /// Are we big-endian or little-endian?
  /// https://stackoverflow.com/a/1001328
  int endianness_test = 1;
  bool i_am_little_endian = (*(char*)&endianness_test == 1);

  /// Publish result
  flow_img.height       = img_0.rows;
  flow_img.width        = img_0.cols;
  flow_img.encoding     = sensor_msgs::image_encodings::TYPE_32FC1; //=="32FC1"
  flow_img.is_bigendian = (not i_am_little_endian);
  flow_img.step         = flow_img.width*4;

  size_t size = 2*flow_img.step*flow_img.height;
  flow_img.data.resize(size);
  std::memcpy(&flow_img.data[0], &prediction.data[0], size);

  flow_pub_ptr.publish(flow_img);

  /// Set previous image
  if (mode == "adjacent-frames")
    last_img = img_1;

  /// Update status
  processed_frames++;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "flownet");
  ROS_INFO("Node initialized");

  ::google::InitGoogleLogging("LMB_flownet");

  ros::NodeHandle node;

  /// Determine variant
  std::string pkg_path = ros::package::getPath("LMB_flownet");
  std::string variant = "FlowNet-cssR";

  ros::param::get("~flownet_variant", variant);
  ros::param::param<std::string>("~mode", mode, "adjacent-frames");
  ros::param::param<bool>("~do_forward_backward_consistency", 
                            do_forward_backward_consistency, false);
  ros::param::param<float>("~FBC_pixel_difference_threshold",
                             FBC_pixel_difference_threshold, 2.0f);
  ros::param::param<bool>("~do_match_feasibility_check", 
                            do_match_feasibility_check, false);

  string model_file   = pkg_path + "/data/" + variant + "/deploy.prototxt";
  string weights_file = pkg_path + "/data/" + variant + "/weights.caffemodel.h5";

  ROS_INFO("model_file = %s", model_file.c_str());
  ROS_INFO("weights_file = %s", weights_file.c_str());

  classifier_ptr = new NetWrapper(model_file, weights_file);
  ROS_INFO("Network created");

  /// Output topics
  ROS_INFO("Advertising flow topic");
  flow_pub_ptr = node.advertise<sensor_msgs::Image>("flownet", 5);

  ROS_INFO("Registering callback");
  ros::Subscriber sub = node.subscribe(node.resolveName("uvc_cam1_rect_mono"), 5, inputImage_callback);

  ROS_INFO("Entering spin()");
  ros::spin();

  /// Tidy up
  ROS_INFO("Exiting...");
  ros::shutdown();
  delete classifier_ptr;

  /// Bye!
  return EXIT_SUCCESS;
}

