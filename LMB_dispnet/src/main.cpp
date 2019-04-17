/// Nikolaus Mayer, 2019

/// System/STL
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
/// Google
#include <glog/logging.h>
/// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
/// ROS
#include <ros/ros.h>
#include <ros/package.h>  // finds package paths
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/fill_image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "stereo_msgs/DisparityImage.h"
/// Caffe
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include <CImg.h>
using namespace cimg_library;

using namespace caffe;

sensor_msgs::Image img_l;
sensor_msgs::Image img_r;
sensor_msgs::CameraInfo info;
std::mutex mutex__dispnet_is_active;
enum class DispNet_status_t {
  WAITING,
  NEW_DATA_AVAILABLE,
  RUNNING,
};
DispNet_status_t dispnet_status{DispNet_status_t::WAITING};
std::vector<ros::Time> backlog;

std::string treatment_of_unprocessed_frames{"none"};

std::string model_file;
std::string trained_file;

/**
 * @brief Replace all occurrences of "from" in "str" with "to"
 * @param str String in which to perform replacement
 * @param from String to look for
 * @param to String with which to replace occurrences of "from"
 *
 * Source: https://stackoverflow.com/a/3418285
 */
void replaceAll(std::string& str, 
                const std::string& from, 
                const std::string& to) 
{
  if(from.empty())
    return;
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}


class NetWrapper {
  public:
  /**
   * @brief Constructor
   * @param model_file Prototxt file describing the graph structure of the
   *                   neural network
   * @param trained_file Protobuf or H5 file containing the trained network
   *                     parameter sets (the "weights")
   */
  NetWrapper(
        const std::string& model_file,
        const std::string& trained_file);

  ~NetWrapper();

  /**
   * @brief Easier access to network inputs
   * @param input_channels Empty; will get one element for each input channel
   * @param index Index of the input_layer to map: 0 or 1
   */
  void WrapInputLayer(
        std::vector<cv::Mat>* input_channels,
        int index);

  /**
   * @brief Estimate pixelwise disparities for a stereo pair
   * @param img_l Left input image ("left" meaning that every object in
   *              the left view must be shifted to the LEFT to find it
   *              somewhere in the right view; NEVER to the RIGHT)
   * @param img_r Right input image
   * @param output Output parameter; disparity map for left image
   */
  void Predict(
        const cv::Mat& img_l, 
        const cv::Mat& img_r,
        cv::Mat& output);
  
  /// The neural network
  Net<float>* net_;
  /// Size of the network's inputs and outputs
  cv::Size input_geometry_;
  bool negate_output;
  /// Filters and consistency checks (output postprocessing)
  bool do_left_right_consistency;
  bool do_mirror_consistency;
  float LRC_pixel_difference_threshold;
  bool do_match_feasibility_check;
  bool do_dispmap_gradient_filter;
  float DGF_pixel_threshold;
  bool do_median_smoothing;
  /// Hard switch to mirror+swap left and right view. This makes the right view
  /// the "reference" for which a disparity map is computed.
  bool right_camera_is_reference;
  /// Path template for writing results to disk
  std::string save_results_filepath_template;
  /// Image crop dimensions
  int _crop_N, _crop_E, _crop_S, _crop_W;
  /// Very small disparities can be a problem. Artifically shifting the stereo
  /// pair adds a constant value to all (correct) disparities which the DispNet
  /// maybe able to deal with better.
  int extra_shift_hack;
};

NetWrapper::NetWrapper(const string& model_file,
                       const string& trained_file) 
: negate_output(false),
  do_left_right_consistency(false),
  do_mirror_consistency(false),
  LRC_pixel_difference_threshold(1.5f),
  do_match_feasibility_check(false),
  do_dispmap_gradient_filter(false),
  DGF_pixel_threshold(1.0f),
  do_median_smoothing(false),
  right_camera_is_reference(false),
  save_results_filepath_template(""),
  _crop_N(0), _crop_E(0), _crop_S(0), _crop_W(0),
  extra_shift_hack(0)
{
  /// We must use GPU mode as the DispNet does not have a working
  /// CPU-only implementation
  Caffe::set_mode(Caffe::GPU);

  /// Create network
  net_ = new Net<float>(model_file, TEST);
  /// Initialize network from pretrained weights
  if (trained_file != "")
    net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs.";

  input_geometry_ = cv::Size(net_->input_blobs()[0]->width(),
                             net_->input_blobs()[0]->height());

  /// Use ROS parameters
  {
    bool v;
    if (ros::param::get("~negate_output", v)) {
      if (v) {
        ROS_INFO("Outputs will be negated");
      }
      negate_output = v;
    }
    if (ros::param::get("~do_left_right_consistency", v)) {
      if (v) {
        ROS_INFO("Enabling left-right-consistency check");
      }
      do_left_right_consistency = v;
    }
    if (ros::param::get("~do_left_right_consistency", v)) {
      if (v) {
        ROS_INFO("Enabling left-right-consistency check");
      }
      do_left_right_consistency = v;
    }
    if (ros::param::get("~do_mirror_consistency", v)) {
      if (v) {
        ROS_INFO("Enabling mirror-consistency check");
      }
      do_mirror_consistency = v;
    }
    if (ros::param::get("~do_match_feasibility_check", v)) {
      if (v) {
        ROS_INFO("Enabling disparity match feasibility check");
      }
      do_match_feasibility_check = v;
    }
    if (ros::param::get("~do_dispmap_gradient_filter", v)) {
      if (v) {
        ROS_INFO("Enabling disparity output gradient filter");
      }
      do_dispmap_gradient_filter = v;
    }
    if (ros::param::get("~do_median_smoothing", v)) {
      if (v) {
        ROS_INFO("Enabling disparity map median-filter smoothing");
      }
      do_median_smoothing = v;
    }
    if (ros::param::get("~right_camera_is_reference", v)) {
      if (v) {
        ROS_INFO("Switching to right-camera-is-reference-camera mode");
      }
      right_camera_is_reference = v;
    }
  }
  {
    float v;
    if (ros::param::get("~LRC_pixel_difference_threshold", v)) {
      ROS_INFO("Setting left-right-consistency threshold: %f px", v);
      LRC_pixel_difference_threshold = v;
    }
    if (ros::param::get("~DGF_pixel_threshold", v)) {
      ROS_INFO("Setting disparity gradient filter threshold: %f px", v);
      DGF_pixel_threshold = v;
    }
  }
  {
    std::string v;
    if (ros::param::get("~save_results_filepath_template", v)) {
      ROS_INFO("Saving disparity outputs: '%s'", v.c_str());
      save_results_filepath_template = v;
    }
  }
  {
    std::vector<int> tmp;
    if (ros::param::get("~crop__north_east_south_west", tmp)) {
      if (tmp.size() == 4) {
        _crop_N = tmp[0];
        _crop_E = tmp[1];
        _crop_S = tmp[2];
        _crop_W = tmp[3];
        ROS_INFO("Using image crop (North-East-South-West): %d, %d, %d, %d", 
                 _crop_N, _crop_E, _crop_S, _crop_W);
      }
    }
  }
  {
    int v;
    if (ros::param::get("~disparity_enhancement_shift", v)) {
      if (v < 0) {
        ROS_INFO("Parameter \"disparity_enhancement_shift\" must be positive");
        v *= -1;
      }
      ROS_INFO("Stereo pair shift: %d px", v);
      extra_shift_hack = v;
    }
  }
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


void NetWrapper::Predict(const cv::Mat& img_l,
                         const cv::Mat& img_r,
                         cv::Mat& output) 
{
  /// feed data
  std::vector<cv::Mat> input_channels_l;
  std::vector<cv::Mat> input_channels_r;
  WrapInputLayer(&input_channels_l, 0);
  WrapInputLayer(&input_channels_r, 1);

  if (right_camera_is_reference) {
    /// Mirror images left-right and swap them
    for (int y = 0; y < input_channels_l[0].rows; ++y) {
      for (int x = 0; x < input_channels_r[0].cols; ++x) {
        input_channels_l[0].at<float>(y,x) = img_r.at<cv::Vec3f>(y,input_channels_r[0].cols-1-x)[0];
        input_channels_l[1].at<float>(y,x) = img_r.at<cv::Vec3f>(y,input_channels_r[0].cols-1-x)[1];
        input_channels_l[2].at<float>(y,x) = img_r.at<cv::Vec3f>(y,input_channels_r[0].cols-1-x)[2];
      }
    }
    CHECK(reinterpret_cast<float*>(input_channels_l[0].data) == 
          net_->input_blobs()[0]->cpu_data());
    for (int y = 0; y < input_channels_l[0].rows; ++y) {
      for (int x = 0; x < input_channels_l[0].cols; ++x) {
        input_channels_r[0].at<float>(y,x) = img_l.at<cv::Vec3f>(y,input_channels_l[0].cols-1-x)[0];
        input_channels_r[1].at<float>(y,x) = img_l.at<cv::Vec3f>(y,input_channels_l[0].cols-1-x)[1];
        input_channels_r[2].at<float>(y,x) = img_l.at<cv::Vec3f>(y,input_channels_l[0].cols-1-x)[2];
      }
    }
  } else {
    for (int y = 0; y < input_channels_l[0].rows; ++y) {
      for (int x = 0; x < input_channels_l[0].cols; ++x) {
        input_channels_l[0].at<float>(y,x) = img_l.at<cv::Vec3f>(y,x)[0];
        input_channels_l[1].at<float>(y,x) = img_l.at<cv::Vec3f>(y,x)[1];
        input_channels_l[2].at<float>(y,x) = img_l.at<cv::Vec3f>(y,x)[2];
      }
    }
    CHECK(reinterpret_cast<float*>(input_channels_l[0].data) == 
          net_->input_blobs()[0]->cpu_data());
    for (int y = 0; y < input_channels_r[0].rows; ++y) {
      for (int x = 0; x < input_channels_r[0].cols; ++x) {
        input_channels_r[0].at<float>(y,x) = img_r.at<cv::Vec3f>(y,x+extra_shift_hack)[0];
        input_channels_r[1].at<float>(y,x) = img_r.at<cv::Vec3f>(y,x+extra_shift_hack)[1];
        input_channels_r[2].at<float>(y,x) = img_r.at<cv::Vec3f>(y,x+extra_shift_hack)[2];
      }
    }
  }

  net_->Forward();

  Blob<float>* output_layer = net_->output_blobs()[net_->num_outputs()-1];
  cv::Mat prediction(input_geometry_.height, input_geometry_.width,
                     CV_32FC1, output_layer->mutable_cpu_data());

  if (right_camera_is_reference) {
    for (int y = 0; y < prediction.rows; ++y) {
      for (int x = 0; x < prediction.cols; ++x) {
        output.at<float>(y,x) = prediction.at<float>(y,prediction.cols-1-x);
      }
    }
  } else {
    for (int y = 0; y < prediction.rows; ++y) {
      for (int x = 0; x < prediction.cols; ++x) {
        output.at<float>(y,x) = prediction.at<float>(y,x);
      }
    }
  }

  if (do_median_smoothing) {
    cv::Mat tmp_median(output);
    cv::medianBlur(tmp_median, output, 5);
  }

  if (negate_output) {
    output *= -1;
  }

  output += extra_shift_hack;
}


NetWrapper* classifier_ptr{nullptr};
ros::Publisher disp_pub_ptr;
ros::Publisher disp_pub_with_dummies_ptr;
//ros::Publisher lr_disp_pub_ptr;


/**
 * @brief Feed image data to the neural network, retrieve the
 *        result, and publishe a disparity map.
 */
void run_dispnet()
{
  //sensor_msgs::ImageConstPtr img_msg_l{&img_l};
  //sensor_msgs::ImageConstPtr img_msg_r{&img_r};
  //sensor_msgs::CameraInfoConstPtr caminfo_msg_l{&info};
  /// DO NOT "delete" these pointers
  sensor_msgs::Image* img_l_copy = new sensor_msgs::Image(img_l);
  sensor_msgs::Image* img_r_copy = new sensor_msgs::Image(img_r);
  sensor_msgs::CameraInfo* info_copy = new sensor_msgs::CameraInfo(info);
  sensor_msgs::ImageConstPtr img_msg_l{img_l_copy};
  sensor_msgs::ImageConstPtr img_msg_r{img_r_copy};
  sensor_msgs::CameraInfoConstPtr caminfo_msg_l{info_copy};

  int crop_N = static_cast<int>(caminfo_msg_l->P[0]);
  int crop_E = static_cast<int>(caminfo_msg_l->P[1]);
  int crop_S = static_cast<int>(caminfo_msg_l->P[2]);
  int crop_W = static_cast<int>(caminfo_msg_l->P[3]);
  if (ros::param::has("~crop__north_east_south_west")) {
    crop_N = classifier_ptr->_crop_N;
    crop_E = classifier_ptr->_crop_E;
    crop_S = classifier_ptr->_crop_S;
    crop_W = classifier_ptr->_crop_W;
  }

  /// Copy data from left image message. This can be done way easier,
  /// but cv::Mat::clone is broken on my dev machine (version conflicts?)
  cv_bridge::CvImagePtr cv_ptr_l =
      cv_bridge::toCvCopy(img_msg_l, sensor_msgs::image_encodings::BGR8);
  cv::Mat img_l(cv_ptr_l->image.rows, cv_ptr_l->image.cols, CV_32FC3);
  for (int y = 0; y < cv_ptr_l->image.rows; ++y) {
    for (int x = 0; x < cv_ptr_l->image.cols; ++x) {
      img_l.at<cv::Vec3f>(y,x)[0] = cv_ptr_l->image.at<cv::Vec3b>(y,x)[0];
      img_l.at<cv::Vec3f>(y,x)[1] = cv_ptr_l->image.at<cv::Vec3b>(y,x)[1];
      img_l.at<cv::Vec3f>(y,x)[2] = cv_ptr_l->image.at<cv::Vec3b>(y,x)[2];
    }
  }

  /// Copy data from right image message
  cv_bridge::CvImagePtr cv_ptr_r =
      cv_bridge::toCvCopy(img_msg_r, sensor_msgs::image_encodings::BGR8);
  cv::Mat img_r(cv_ptr_r->image.rows, cv_ptr_r->image.cols, CV_32FC3);
  for (int y = 0; y < cv_ptr_r->image.rows; ++y) {
    for (int x = 0; x < cv_ptr_r->image.cols; ++x) {
      img_r.at<cv::Vec3f>(y,x)[0] = cv_ptr_r->image.at<cv::Vec3b>(y,x)[0];
      img_r.at<cv::Vec3f>(y,x)[1] = cv_ptr_r->image.at<cv::Vec3b>(y,x)[1];
      img_r.at<cv::Vec3f>(y,x)[2] = cv_ptr_r->image.at<cv::Vec3b>(y,x)[2];
    }
  }

  if (classifier_ptr) {
    /// The raw input images contain undistortion and rectification
    /// artifacts. The neural network's config file is hardcoded to a
    /// specific input resolution (can be changed), so we crop the
    /// inputs here. This means that we must later adjust the outputs
    /// such that the "disparity" values are valid for the original,
    /// uncropped resolution.
    cv::Mat img_l_cropped(classifier_ptr->input_geometry_.height, 
                          classifier_ptr->input_geometry_.width, CV_32FC3);
    cv::Mat img_r_cropped(classifier_ptr->input_geometry_.height, 
                          classifier_ptr->input_geometry_.width, CV_32FC3);
    for (int y = 0; y < img_l_cropped.rows; ++y) {
      for (int x = 0; x < img_l_cropped.cols; ++x) {
        img_l_cropped.at<cv::Vec3f>(y,x)[0] = img_l.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[0];
        img_l_cropped.at<cv::Vec3f>(y,x)[1] = img_l.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[1];
        img_l_cropped.at<cv::Vec3f>(y,x)[2] = img_l.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[2];

        img_r_cropped.at<cv::Vec3f>(y,x)[0] = img_r.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[0];
        img_r_cropped.at<cv::Vec3f>(y,x)[1] = img_r.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[1];
        img_r_cropped.at<cv::Vec3f>(y,x)[2] = img_r.at<cv::Vec3f>(y+crop_N,
                                                                  x+crop_W)[2];
      }
    }

    /// Image -> NeuralNetwork -> Result
    cv::Mat prediction(classifier_ptr->input_geometry_.height, 
                       classifier_ptr->input_geometry_.width, CV_32FC1);
    classifier_ptr->Predict(img_l_cropped, img_r_cropped, prediction);


    /// Estimate disparity map for rotated image pair, to enable left-
    /// right consistency checking
    cv::Mat lr_prediction(prediction.rows,prediction.cols,CV_32FC1);
    if (classifier_ptr->do_left_right_consistency) {
      cv::Mat lr_img_l_cropped(img_l_cropped);
      cv::Mat lr_img_r_cropped(img_r_cropped);
      cv::flip(img_l_cropped, lr_img_l_cropped, -1);
      cv::flip(img_r_cropped, lr_img_r_cropped, -1);
      cv::Mat tmp(prediction.rows,prediction.cols,CV_32FC1);
      classifier_ptr->Predict(lr_img_r_cropped, lr_img_l_cropped, tmp);
      cv::flip(tmp, lr_prediction, -1);
      
      ///stereo_msgs::DisparityImage disp_msg;
      ///disp_msg.header.stamp = img_msg_l->header.stamp;
      ///{
      ///  /// Are we big-endian or little-endian?
      ///  /// https://stackoverflow.com/a/1001328
      ///  int endianness_test = 1;
      ///  bool i_am_little_endian = (*(char*)&endianness_test == 1);

      ///  sensor_msgs::Image disp_img;
      ///  disp_img.height       = img_l.rows;
      ///  disp_img.width        = img_l.cols;
      ///  disp_img.encoding     = sensor_msgs::image_encodings::TYPE_32FC1;
      ///  disp_img.is_bigendian = (not i_am_little_endian);
      ///  disp_img.step         = disp_img.width*4;
      ///  disp_img.data.resize(disp_img.step*disp_img.height);
      ///  cv::Mat tmp(img_l.rows, img_l.cols, CV_32FC1);
      ///  tmp = -1.f;
      ///  for (int y = 0; y < lr_prediction.rows; ++y) {
      ///    for (int x = 0; x < lr_prediction.cols; ++x) {
      ///      tmp.at<float>(y+crop_N,x+crop_W) = -1*lr_prediction.at<float>(y,x);
      ///    }
      ///  }
      ///  std::memcpy(&disp_img.data[0], &tmp.data[0], 
      ///              disp_img.step*disp_img.height);

      ///  disp_msg.image = disp_img;
      ///}
      ///disp_msg.f             = caminfo_msg_l->K[0];
      ///disp_msg.T             = caminfo_msg_l->P[4]; /// 0.1f;
      ///disp_msg.min_disparity = 0.f;
      ///disp_msg.max_disparity = 100000.f;
      ///disp_msg.delta_d       = 0.f;
      ///lr_disp_pub_ptr.publish(disp_msg);
    }

    /// Estimate disparity map for mirrored image pair, to enable mirror
    /// consistency checking
    cv::Mat mirror_prediction(prediction.rows,prediction.cols,CV_32FC1);
    cv::Mat mirror_lr_prediction(prediction.rows,prediction.cols,CV_32FC1);
    if (classifier_ptr->do_mirror_consistency) {
      cv::Mat mirror_img_l_cropped(img_l_cropped);
      cv::Mat mirror_img_r_cropped(img_r_cropped);
      cv::flip(img_l_cropped, mirror_img_l_cropped, 0);
      cv::flip(img_r_cropped, mirror_img_r_cropped, 0);
      cv::Mat tmp(prediction.rows,prediction.cols,CV_32FC1);
      classifier_ptr->Predict(mirror_img_r_cropped, mirror_img_l_cropped, tmp);
      cv::flip(tmp, mirror_prediction, 1);

      if (classifier_ptr->do_left_right_consistency) {
        cv::Mat mirror_lr_img_l_cropped(mirror_img_r_cropped);
        cv::Mat mirror_lr_img_r_cropped(mirror_img_l_cropped);
        cv::flip(mirror_img_l_cropped, mirror_lr_img_r_cropped, -1);
        cv::flip(mirror_img_r_cropped, mirror_lr_img_l_cropped, -1);
        classifier_ptr->Predict(mirror_lr_img_r_cropped, 
                                mirror_lr_img_l_cropped, 
                                tmp);
        cv::flip(tmp, mirror_lr_prediction, 0);
      }
    }


    /// Publish disparity map
    {
      stereo_msgs::DisparityImage disp_msg;
      disp_msg.header.stamp = img_msg_l->header.stamp;
      disp_msg.header.seq = img_msg_l->header.seq;
      
      /// Are we big-endian or little-endian?
      /// https://stackoverflow.com/a/1001328
      int endianness_test = 1;
      bool i_am_little_endian = (*(char*)&endianness_test == 1);

      sensor_msgs::Image disp_img;
      disp_img.height       = img_l.rows;
      disp_img.width        = img_l.cols;
      disp_img.encoding     = sensor_msgs::image_encodings::TYPE_32FC1;
      disp_img.is_bigendian = (not i_am_little_endian);
      disp_img.step         = disp_img.width*4;
      disp_img.data.resize(disp_img.step*disp_img.height);
      cv::Mat tmp(img_l.rows, img_l.cols, CV_32FC1);
      tmp = -1.f;
      for (int y = 0; y < prediction.rows; ++y) {
        for (int x = 0; x < prediction.cols; ++x) {
          tmp.at<float>(y+crop_N,x+crop_W) = -1.f*prediction.at<float>(y,x);
        }
      }

      /// Left-right consistency check
      if (classifier_ptr->do_left_right_consistency) {
        for (int y = crop_N; y < tmp.rows-crop_S; ++y) {
          for (int x = crop_W; x < tmp.cols-crop_E; ++x) {
            const float l = tmp.at<float>(y,x);
            const float other_pos = (classifier_ptr->right_camera_is_reference 
                                     ? x+l 
                                     : x-l);
            const float r = -1.f*lr_prediction.at<float>(y-crop_N, other_pos-crop_W);
            if (std::abs(l-r)>classifier_ptr->LRC_pixel_difference_threshold){
              tmp.at<float>(y,x) = 0.f;
            }
          }
        }
      }

      /// Mirror consistency check
      if (classifier_ptr->do_mirror_consistency) {
        for (int y = crop_N; y < tmp.rows-crop_S; ++y) {
          for (int x = crop_W; x < tmp.cols-crop_E; ++x) {
            const float l = tmp.at<float>(y,x);
            if (l == -1.f)
              continue;
            const float other_pos = (classifier_ptr->right_camera_is_reference 
                                     ? x+l 
                                     : x-l);
            const float r = -1.f*mirror_prediction.at<float>(y, other_pos);
            if (std::abs(l-r)>classifier_ptr->LRC_pixel_difference_threshold){
              tmp.at<float>(y,x) = 0.f;
            }
          }
        }
      }

      /// Left-right-and-mirror consistency check
      if (classifier_ptr->do_left_right_consistency and
          classifier_ptr->do_mirror_consistency) {
        for (int y = crop_N; y < tmp.rows-crop_S; ++y) {
          for (int x = crop_W; x < tmp.cols-crop_E; ++x) {
            const float l = tmp.at<float>(y,x);
            if (l == -1.f)
              continue;
            const float r = -1.f*mirror_lr_prediction.at<float>(y,x);
            if (std::abs(l-r)>classifier_ptr->LRC_pixel_difference_threshold){
              tmp.at<float>(y,x) = 0.f;
            }
          }
        }
      }
      
      /// Pixel without true match (at estimated disparity, the
      /// correspondence would be outside the image bounds)
      if (classifier_ptr->do_match_feasibility_check) {
        for (int y = crop_N; y < tmp.rows-crop_S; ++y) {
          for (int x = crop_W; x < tmp.cols-crop_E; ++x) {
            const float l = tmp.at<float>(y,x);
            if (classifier_ptr->right_camera_is_reference) {
              if (x+l < 0 or x+l >= tmp.cols) {
                tmp.at<float>(y,x) = 0.f;
              }
            } else {
              if (x-l < 0 or x-l >= tmp.cols) {
                tmp.at<float>(y,x) = 0.f;
              }
            }
          }
        }
      }

      /// Filter out pixels along depth discontinuities. This
      /// helps against the DispNet's "curtain" artifacts
      if (classifier_ptr->do_dispmap_gradient_filter) {
        for (int y = crop_N; y < tmp.rows-crop_S; ++y) {
          for (int x = crop_W; x < tmp.cols-crop_E; ++x) {
            const float l = prediction.at<float>(y-crop_N,x-crop_W);
            const float dd_threshold = classifier_ptr->DGF_pixel_threshold;
            if (((x > 0) and 
                 (std::abs(l - prediction.at<float>(y-crop_N,x-crop_W-1)) 
                  > dd_threshold)) or
                ((y > 0) and 
                 (std::abs(l - prediction.at<float>(y-crop_N-1,x-crop_W)) 
                  > dd_threshold)) or
                ((x < prediction.cols-1) and 
                 (std::abs(l - prediction.at<float>(y-crop_N,x-crop_W+1)) 
                  > dd_threshold)) or
                ((y < prediction.rows-1) and 
                 (std::abs(l - prediction.at<float>(y-crop_N+1,x-crop_W)) 
                  > dd_threshold))) {
              tmp.at<float>(y,x) = 0.f;
            }
          }
        }
      }

      std::memcpy(&disp_img.data[0], &tmp.data[0], 
                  disp_img.step*disp_img.height);

      disp_msg.image = disp_img;
      
      disp_msg.f             = caminfo_msg_l->K[0];
      disp_msg.T             = -1.f*caminfo_msg_l->P[4]; /// 0.1f;
      disp_msg.min_disparity = 0.f;
      disp_msg.max_disparity = 100000.f;
      disp_msg.delta_d       = 0.f;
      disp_pub_ptr.publish(disp_msg);
      if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
        stereo_msgs::DisparityImage disp_msg_copy = disp_msg;
        disp_pub_with_dummies_ptr.publish(disp_msg_copy);
      }


      /// Write disparity map to file
      if (classifier_ptr->save_results_filepath_template != "") {
        CImg<float> output_pfm(tmp.cols, tmp.rows, 1, 1);
        output_pfm.fill(0.f);
        cimg_forXY(output_pfm, x, y) {
          output_pfm(x, y) = tmp.at<float>(y, x);
        }

        static int file_counter{0};

        char output_path_buffer[1024];
        std::sprintf(output_path_buffer, 
                     classifier_ptr->save_results_filepath_template.c_str(),
                     file_counter);
        output_pfm.save_pfm(output_path_buffer);
        ++file_counter;
      }
    }
  }
}


void run_dispnet__worker_loop()
{
  /// With multithreading, the thread using the network must initialize it
  classifier_ptr = new NetWrapper(model_file, trained_file);
  
  /// Deferred lock (not yet locked)
  while (ros::ok()) {
    if (dispnet_status != DispNet_status_t::NEW_DATA_AVAILABLE) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    dispnet_status = DispNet_status_t::RUNNING;
    run_dispnet();
    dispnet_status = DispNet_status_t::WAITING;
  }

  if (classifier_ptr)
    delete classifier_ptr;
}



/**
 * @brief Callback for synchronized stereo image topics. This method
 *        feeds the image data to the neural network, retrieves the
 *        result, and publishes a disparity map.
 * @param img_msg_l Left image topic message (see documentation of
 *                  NetWrapper::Predict for how "left" is defined)
 * @param img_msg_r Right image topic message
 * @param caminfo_msg_l Left camera's info message
 */
void inputImages_callback(const sensor_msgs::ImageConstPtr& img_msg_l,
                          const sensor_msgs::ImageConstPtr& img_msg_r,
                          const sensor_msgs::CameraInfoConstPtr& caminfo_msg_l)
{
  if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
    if (dispnet_status != DispNet_status_t::WAITING) {
      /// DispNet is busy; push callback into backlog
      backlog.push_back(img_msg_l->header.stamp);
      //ROS_INFO("Backlog now at %d frames", backlog.size());
      return;
    }

    /// Send dummy disparity maps
    for (size_t i = 0; i < backlog.size(); ++i) {
      stereo_msgs::DisparityImage dummy_disp_msg;
      dummy_disp_msg.header.stamp = backlog[i];
      dummy_disp_msg.image.width  = 0;
      dummy_disp_msg.image.height = 0;
      disp_pub_with_dummies_ptr.publish(dummy_disp_msg);
      //ROS_INFO("Draining backlog: %d/%d", backlog.size()-i, backlog.size());
    }
    backlog.clear();
  }
  

  img_l = *img_msg_l;
  img_r = *img_msg_r;
  info  = *caminfo_msg_l;
  /// Run DispNet or notify worker thread that it can run
  if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
    /// Set flag and wait until worker thread has unset the flag
    dispnet_status = DispNet_status_t::NEW_DATA_AVAILABLE;
    while (dispnet_status != DispNet_status_t::RUNNING) {
      if (not ros::ok())
        return;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } else {
    run_dispnet();
  }
}

void inputImages_callback_dummy(const sensor_msgs::ImageConstPtr& img_msg_l,
                                const sensor_msgs::ImageConstPtr& img_msg_r)
{
  sensor_msgs::CameraInfo::Ptr dummy_ptr(new sensor_msgs::CameraInfo);
  inputImages_callback(img_msg_l, img_msg_r, dummy_ptr);
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "dispnet");
  ROS_INFO("Node initialized");
  
  ros::NodeHandle node;

  std::string pkg_path = ros::package::getPath("LMB_dispnet");

  ::google::InitGoogleLogging("LMB_dispnet");
  {
    std::string s;
    if (ros::param::get("~weights_file", s)) {
      ROS_INFO("Using CNN weights file \"%s\"", s.c_str());
      trained_file = pkg_path + "/data/" + s;
    }
  }


  bool use_caminfo_topic{false};
  if (ros::param::get("~with_caminfo_topic", use_caminfo_topic)) {
    if (use_caminfo_topic) {
      ROS_INFO("Using camera_info topic data");
    }
  }

  {
    int in_w, in_h, re_w, re_h;
    float resample;

    int shift{0};
    {
      if (ros::param::get("~disparity_enhancement_shift", shift)) {
        if (shift < 0) {
          shift *= -1;
        }
      }
    }

    /// Compute parameter values
    if (use_caminfo_topic) {
      if (ros::param::has("~image_dimensions")) {
        ROS_INFO("Parameter \"image_dimensions\" is set, but will be ignored "
                 " because \"use_caminfo_topic\" is active.");
      }
      if (ros::param::has("~crop__north_east_south_west")) {
        ROS_INFO("Parameter \"crop__north_east_south_west\" is set, but will "
                 "be ignored because \"use_caminfo_topic\" is active.");
      }

      std::string caminfo_topic = node.resolveName("/uvc_cam1_rect_mono/camera_info");
      const sensor_msgs::CameraInfoConstPtr caminfo_msg_ptr = 
          ros::topic::waitForMessage<sensor_msgs::CameraInfo>(caminfo_topic, node);
      in_w = caminfo_msg_ptr->width  - caminfo_msg_ptr->P[1]
                                     - caminfo_msg_ptr->P[3] - shift;
      in_h = caminfo_msg_ptr->height - caminfo_msg_ptr->P[0]
                                     - caminfo_msg_ptr->P[2];
      re_w = static_cast<int>(std::ceil(in_w/64.)*64);
      re_h = static_cast<int>(std::ceil(in_h/64.)*64);
      resample = static_cast<float>(in_w)/re_w;

      /// Debug output
      {
        std::ostringstream oss;
        oss << "Remapping (" << in_w << "," << in_h << ") "
            <<        "to (" << re_w << "," << re_h << "), "
            << ", scaling factor " << resample;
        if (shift > 0) oss << ", shift " << shift << " px";
        oss << std::endl
            << "Croppings are"
            << " N=" << static_cast<int>(caminfo_msg_ptr->P[0])
            << ",E=" << static_cast<int>(caminfo_msg_ptr->P[1])
            << ",S=" << static_cast<int>(caminfo_msg_ptr->P[2])
            << ",W=" << static_cast<int>(caminfo_msg_ptr->P[3]);
        ROS_INFO(oss.str().c_str());
      }
    } else {
      std::vector<int> preset_dims{752, 480};
      if (ros::param::get("~image_dimensions", preset_dims)) {
        if (preset_dims.size() == 2) {
          std::ostringstream oss;
          oss << "Using preset image dimensions: ("
              << preset_dims[0] << ", " << preset_dims[1] << ")";
          ROS_INFO(oss.str().c_str());
        }
      }
      std::vector<int> crop__north_east_south_west;
      if (ros::param::get("~crop__north_east_south_west",
                          crop__north_east_south_west)) {
        if (crop__north_east_south_west.size() == 4) {
          ROS_INFO("Using image crop (N-E-S-W): %d, %d, %d, %d", 
                   crop__north_east_south_west[0],
                   crop__north_east_south_west[1],
                   crop__north_east_south_west[2],
                   crop__north_east_south_west[3]);
        }
      }
      in_w = preset_dims[0] - crop__north_east_south_west[1]
                            - crop__north_east_south_west[3] - shift;
      in_h = preset_dims[1] - crop__north_east_south_west[0]
                            - crop__north_east_south_west[2];
      re_w = static_cast<int>(std::ceil(in_w/64.)*64);
      re_h = static_cast<int>(std::ceil(in_h/64.)*64);
      resample = static_cast<float>(in_w)/re_w;
    }

    if (ros::param::get("~treatment_of_unprocessed_frames",
                        treatment_of_unprocessed_frames)) {
      if        (treatment_of_unprocessed_frames == "none") {
      } else if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
        ROS_INFO("Publishing \"dispnet_dummy\" topic");
      } else {
        ROS_WARN("Unknown value \"%s\" for parameter \"%s\"; using \"none\"",
                 treatment_of_unprocessed_frames.c_str(),
                 "treatment_of_unprocessed_frames");
        treatment_of_unprocessed_frames = "none";
      }
    }
    
    /// Read template
    std::string model_template_text;
    {
      std::string model_template_path = trained_file + ".prototxt.template";
      std::ifstream model_template(model_template_path.c_str());
      if (model_template.bad() or not model_template.is_open()) {
        throw std::runtime_error("Could not open model template file");
      }
      model_template.seekg(0, std::ios::end);
      model_template_text.resize(model_template.tellg());
      model_template.seekg(0, std::ios::beg);
      model_template.read(&model_template_text[0], model_template_text.size());
    }

    /// Fill template
    std::string model_text{model_template_text};
    {
      std::ostringstream oss;
      oss << in_w;
      replaceAll(model_text, "%(in_width)d", oss.str());

      oss.str("");
      oss << in_h;
      replaceAll(model_text, "%(in_height)d", oss.str());

      oss.str("");
      oss << re_w;
      replaceAll(model_text, "%(resample_width)d", oss.str());

      oss.str("");
      oss << re_h;
      replaceAll(model_text, "%(resample_height)d", oss.str());

      oss.str("");
      oss << resample;
      replaceAll(model_text, "%(resample_factor)f", oss.str());
    }
    const int model_text_size = model_text.size();
    
    /// Write template
    {
      std::string model_output_path = pkg_path+"/data/model.prototxt.tmp";
      std::ofstream model_output(model_output_path.c_str());
      if (model_output.bad() or not model_output.is_open()) {
        throw std::runtime_error("Could not write model file");
      }
      model_output.write(&model_text[0], model_text_size);
    }
  }

  ROS_INFO("Network created");

  /// Output topics
  ROS_INFO("Advertising disparity topic");
  disp_pub_ptr = node.advertise<stereo_msgs::DisparityImage>("dispnet", 5);
  if (treatment_of_unprocessed_frames == "publish_dummy_disparities")
    disp_pub_with_dummies_ptr = node.advertise<stereo_msgs::DisparityImage>("dispnet_dummy", 5);

  
  /// Start worker thread
  std::thread* dispnet_worker_thread_ptr{nullptr};
  if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
    ROS_INFO("Starting DispNet worker thread");
    dispnet_worker_thread_ptr = new std::thread(run_dispnet__worker_loop);
  } else {
    classifier_ptr = new NetWrapper(model_file, trained_file);
  }


  /// Input topics
  ROS_INFO("Registering sync'ed image streams");
  ros::TransportHints my_hints{ros::TransportHints().tcpNoDelay(true)};
  if (use_caminfo_topic) {
    std::string topic_cam_l = node.resolveName("/uvc_cam1_rect_mono");
    std::string topic_cam_r = node.resolveName("/uvc_cam0_rect_mono");
    std::string topic_caminfo_l = node.resolveName("/uvc_cam1_rect_mono/camera_info");
    message_filters::Subscriber<sensor_msgs::Image> cam_l_sub(node,
                                                    topic_cam_l, 5,
                                                    my_hints);
    message_filters::Subscriber<sensor_msgs::Image> cam_r_sub(node,
                                                    topic_cam_r, 5,
                                                    my_hints);
    message_filters::Subscriber<sensor_msgs::CameraInfo> caminfo_l_sub(node, 
                                                    topic_caminfo_l, 5,
                                                    my_hints);
    message_filters::TimeSynchronizer<sensor_msgs::Image, 
                                      sensor_msgs::Image,
                                      sensor_msgs::CameraInfo> sync(
        cam_l_sub, cam_r_sub, caminfo_l_sub, 5);
    sync.registerCallback(boost::bind(&inputImages_callback, _1, _2, _3));

    ros::spin();
  } else {
    std::string topic_cam_l = node.resolveName("/uvc_cam1_rect_mono");
    std::string topic_cam_r = node.resolveName("/uvc_cam0_rect_mono");
    message_filters::Subscriber<sensor_msgs::Image> cam_l_sub(node,
                                                    topic_cam_l, 5,
                                                    my_hints);
    message_filters::Subscriber<sensor_msgs::Image> cam_r_sub(node,
                                                    topic_cam_r, 5,
                                                    my_hints);
    message_filters::TimeSynchronizer<sensor_msgs::Image, 
                                      sensor_msgs::Image> sync(
        cam_l_sub, cam_r_sub, 5);
    sync.registerCallback(boost::bind(&inputImages_callback_dummy, _1, _2));

    ros::spin();
  }


  /// Tidy up
  if (treatment_of_unprocessed_frames == "publish_dummy_disparities") {
    if (dispnet_worker_thread_ptr and dispnet_worker_thread_ptr->joinable()) {
      ROS_INFO("Starting DispNet worker thread");
      dispnet_worker_thread_ptr->join();
      delete dispnet_worker_thread_ptr;
    }
  }
  ROS_INFO("Exiting...");
  ros::shutdown();
  if (classifier_ptr)
    delete classifier_ptr;

  /// Bye!
  return EXIT_SUCCESS;
}

