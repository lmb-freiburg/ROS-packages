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
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
/// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/PointIndices.h>

typedef pcl::PointXYZRGB Point_6;
typedef pcl::PointCloud<Point_6> PCL_6;


/// Node parameters
std::string mode{"adjacent-frames"};
bool invert_cx{false};
bool invert_cy{false};
int verbosity{0};

/// Camera intrinsics matrix and their inverse
Eigen::Matrix3f K_left, Ki_left;

/// Publishers for this node's topics
ros::Publisher republish_left_camerainfo;
ros::Publisher republish_left;
ros::Publisher republish_right;
ros::Publisher pcl_pub;
ros::Publisher arrows_pub;
ros::Publisher sceneflow_img_vis;

/// Placeholders for input data and DispNet/FlowNet outputs
stereo_msgs::DisparityImage previous_disparity, 
                            current_disparity;
sensor_msgs::Image previous_left,
                   previous_right,
                   current_left,
                   current_right,
                   left_flow;

/// Program status flags
bool dispnet_is_busy{false};
bool flownet_is_busy{false};
bool ok_for_next_round{true};
bool first_frame{true};



/**
 * Main callback
 * Parameters:
 * img_msg_l left-view image
 * img_msg_r right-view image
 * caminfo_msg_l left-view camera info
 */
void callback(const sensor_msgs::ImageConstPtr& img_msg_l,
              const sensor_msgs::ImageConstPtr& img_msg_r,
              const sensor_msgs::CameraInfoConstPtr& caminfo_msg_l)
{
  /// No callbacks why processing
  if (dispnet_is_busy or flownet_is_busy or not ok_for_next_round)
    return;
  if (verbosity >= 1)
    ROS_INFO("Got input callback");

  /// Initialize camera intrinsics
  if (K_left(0,0) == 0.f) {
    auto tmpK = caminfo_msg_l->K;
    K_left << tmpK[0], tmpK[1], tmpK[2],
              tmpK[3], tmpK[4], tmpK[5],
              tmpK[6], tmpK[7], tmpK[8];
    if (invert_cx)
      K_left(0, 2) = img_msg_l->width - 1 - K_left(0, 2);
    if (invert_cy)
      K_left(1, 2) = img_msg_l->height - 1 - K_left(1, 2);
    Ki_left << K_left.inverse();
  }

  /// Set status flags
  dispnet_is_busy = true;
  flownet_is_busy = true;
  ok_for_next_round = false;

  /// In "back to back" mode, adjacent frames are used.
  /// In "to first frame" mode, the reference frame is always the initial frame.
  if (mode == "adjacent-frames" or previous_left.width == 0) {
    previous_left  = current_left;
    previous_right = current_right;
  }
  current_left   = *img_msg_l;
  current_right  = *img_msg_r;

  /// Send out data to DispNet/FlowNet
  republish_left.publish(img_msg_l);
  republish_right.publish(img_msg_r);
  republish_left_camerainfo.publish(caminfo_msg_l);
}

/**
 * Callback for DispNet results
 */
void callback_DispNet(const stereo_msgs::DisparityImageConstPtr& disp_msg_ptr)
{
  if (not dispnet_is_busy)
    return;
  if (verbosity >= 1)
    ROS_INFO("Got DispNet callback");

  if (mode == "adjacent-frames" or previous_disparity.image.width == 0)
    previous_disparity = current_disparity;
  current_disparity = *disp_msg_ptr;
  dispnet_is_busy = false;
}

/**
 * Callback for FlowNet results
 */
void callback_FlowNet(const sensor_msgs::ImageConstPtr& flow_msg_ptr)
{
  if (not flownet_is_busy)
    return;
  if (verbosity >= 1)
    ROS_INFO("Got FlowNet callback");

  left_flow = *flow_msg_ptr;
  flownet_is_busy = false;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "sceneflow");
  ros::NodeHandle node;

  /// Parameters
  ros::param::param<std::string>("~mode", mode, "adjacent-frames");
  ros::param::param<bool>("~invert_cx", invert_cx, false);
  ros::param::param<bool>("~invert_cy", invert_cy, false);
  ros::param::param<int>("~verbosity", verbosity, 0);
  
  republish_left_camerainfo = node.advertise<sensor_msgs::CameraInfo>(
                                    "LMB__sceneflow_left/camera_info", 1);
  republish_left            = node.advertise<sensor_msgs::Image>(
                                    "LMB__sceneflow_left", 1);
  republish_right           = node.advertise<sensor_msgs::Image>(
                                    "LMB__sceneflow_right", 1);
  pcl_pub                   = node.advertise<PCL_6>(
                                    "LMB__sceneflow_pcl", 10);
  arrows_pub                = node.advertise<visualization_msgs::MarkerArray>(
                                    "LMB__sceneflow_arrows", 10);
  sceneflow_img_vis         = node.advertise<sensor_msgs::Image>(
                                    "LMB__sceneflow_vis", 1);
  
  /// Subscribe to rectified images (INPUT callback)
  ROS_INFO("Registering sync'ed image streams");
  ros::TransportHints my_hints{ros::TransportHints().tcpNoDelay(true)};
  std::string topic_cam_l = node.resolveName("/uvc_cam1_rect_mono");
  std::string topic_cam_r = node.resolveName("/uvc_cam0_rect_mono");
  std::string topic_caminfo_l = node.resolveName("/uvc_cam1_rect_mono/camera_info");
  message_filters::Subscriber<sensor_msgs::Image> cam_l_sub(node,
                                                  topic_cam_l, 1, my_hints);
  message_filters::Subscriber<sensor_msgs::Image> cam_r_sub(node,
                                                  topic_cam_r, 1, my_hints);
  message_filters::Subscriber<sensor_msgs::CameraInfo> caminfo_l_sub(node, 
                                                  topic_caminfo_l, 1, my_hints);
  message_filters::TimeSynchronizer<sensor_msgs::Image, 
                                    sensor_msgs::Image,
                                    sensor_msgs::CameraInfo> sync(cam_l_sub, 
                                                                  cam_r_sub, 
                                                                  caminfo_l_sub, 
                                                                  1);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  /// DispNet OUTPUT callback
  ros::Subscriber dispnet_subscriber = node.subscribe("dispnet", 100, 
                                                      callback_DispNet);
  /// FlowNet OUTPUT callback
  ros::Subscriber flownet_subscriber = node.subscribe("flownet", 100, 
                                                      callback_FlowNet);

  /// Loop:
  while (ros::ok()) {
    /// Send (left_t, right_t) to DispNet
    /// Send (left_t, left_t+X) to FlowNet
    while (not dispnet_is_busy or not flownet_is_busy) {
      if (not ros::ok()) 
        return EXIT_SUCCESS;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ros::spinOnce();
    }

    if (first_frame) {
      first_frame = false;
      dispnet_is_busy = false;
      flownet_is_busy = false;
      ok_for_next_round = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10000));
      continue;
    }

    /// Wait for synchronized results
    while (dispnet_is_busy or flownet_is_busy) {
      if (not ros::ok()) 
        return EXIT_SUCCESS;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ros::spinOnce();
    }

    ok_for_next_round = true;
  }


  /// Bye!
  return EXIT_SUCCESS;
}

