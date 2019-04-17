/**
 * Nikolaus Mayer, 2019 (mayern@cs.uni-freiburg.de)
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <string>
/// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// ROS
#include <ros/ros.h>
#include <ros/package.h>  // finds package paths
#include <sensor_msgs/Image.h>
#include <sensor_msgs/fill_image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "stereo_msgs/DisparityImage.h"

#include <CImg.h>
using namespace cimg_library;


ros::Publisher publisher__visualized_l;

void inputImages_l_callback(const stereo_msgs::DisparityImageConstPtr& disp_msg_ptr)
{
  ROS_INFO("Callback");

  /// Get images from messages
  CImg<unsigned char> display_image;
  CImg<float> disparity_map;
  {
    cv_bridge::CvImagePtr cv_ptr_disp = cv_bridge::toCvCopy(disp_msg_ptr->image, 
                               sensor_msgs::image_encodings::TYPE_32FC1);
    disparity_map = CImg<float>(cv_ptr_disp->image.cols,
                                cv_ptr_disp->image.rows,1,1);
    cimg_forXY(disparity_map,x,y) {
      disparity_map(x,y) = cv_ptr_disp->image.at<float>(y,x);
    }

    const int W = disparity_map.width();
    const int H = disparity_map.height();

    sensor_msgs::Image depth_map_image;
    depth_map_image.height = H;
    depth_map_image.width = W;
    depth_map_image.encoding = "mono8";
    depth_map_image.step = W;
    depth_map_image.data = std::vector<uint8_t>(W*H);
    const float vmin = disparity_map.min();
    const float vmax = disparity_map.max();
    for (int y = 0; y < disparity_map.height(); ++y) {
      for (int x = 0; x < disparity_map.width(); ++x) {
        float v = disparity_map(x,y);
        v = (v-vmin)/(vmax-vmin)*255;
        v = std::max(0.f, std::min(std::abs(v), 255.f));

        unsigned char v_uchar = static_cast<unsigned char>(v);
        depth_map_image.data.data()[y*W+x] = v_uchar;
      }
    }
    publisher__visualized_l.publish(depth_map_image);

  }
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "VisualizeDisparityImage");
  ROS_INFO("Node initialized");

  ros::NodeHandle node;
  
  publisher__visualized_l = node.advertise<sensor_msgs::Image>("/dispnet_vis", 1);
  
  /// Input topics
  ROS_INFO("Subscribing to disparity image stream");
  std::string topic_disp_l = node.resolveName("dispnet");
  ros::Subscriber image_subscriber_l =
      node.subscribe(topic_disp_l, 1, inputImages_l_callback);

  ros::spin();

  /// Tidy up
  ROS_INFO("Exiting...");
  ros::shutdown();

  /// Bye!
  return EXIT_SUCCESS;
}

