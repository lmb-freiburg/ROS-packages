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
// ROS
#include <ros/ros.h>
#include <ros/package.h>  // finds package paths
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

#include "CImg.h"
using namespace cimg_library;

#include "pacemaker.h"


/**
 * @brief Publish a sensor_msgs::Image message
 * @param img_data Image pixel data
 * @param W Image width
 * @param H Image height
 * @param publisher The topic's publisher
 * @param timestamp Image timestamp
 * @param seq_id Sequence ID
 */
void send_image(
      unsigned char* img_data, 
      int W, 
      int H,
      ros::Publisher& publisher,
      ros::Time timestamp,
      int seq_id)
{
  /// Create output message
  sensor_msgs::Image msg;
  msg.header.stamp = timestamp;
  msg.header.seq = seq_id;
  msg.height = H;
  msg.width = W;
  msg.encoding = "rgb8";
  msg.step = W*3;
  msg.data.resize(W*H*3);

  std::memcpy(&msg.data[0], img_data, W*H*3);

  /// Publish
  publisher.publish(msg);
}

/**
 * @brief Publish a sensor_msgs::CameraInfo message
 * @param W Image width
 * @param H Image height
 * @param K Pointer to 9 values of the K matrix
 * @param publisher The topic's publisher
 * @param timestamp Timestamp
 * @param seq_id Sequence ID
 */
void send_caminfo(
      int W, 
      int H, 
      float* K,
      ros::Publisher& publisher, 
      ros::Time timestamp,
      int seq_id)
{
  sensor_msgs::CameraInfo msg;
  msg.header.stamp = timestamp;
  msg.header.seq = seq_id;
  msg.height = H;
  msg.width = W;
  for (size_t i = 0; i < 9; ++i) { msg.K[i] = K[i]; }
  msg.P[0] = 0;
  msg.P[1] = 0;
  msg.P[2] = 0;
  msg.P[3] = 0;

  publisher.publish(msg);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "images_feeder");
  ROS_INFO("Node initialized");

  ros::NodeHandle node;

  std::string pkg_path = ros::package::getPath("LMB_generic_images_feeder");

  int W, H;

  /// Load images
  std::vector<std::vector<unsigned char*>> image_sets;
  std::vector<std::string> topics;
  std::vector<std::vector<std::string>> images_filepaths;
  std::vector<ros::Publisher> image_publishers;
  std::vector<ros::Publisher> camerainfo_publishers;

  {
    std::vector<std::string> tmp;
    if (ros::param::get("~topic_1", tmp)) {
      if (tmp.size() > 1) {
        topics.push_back(tmp[0]);
        images_filepaths.push_back(std::vector<std::string>{});
        image_sets.push_back(std::vector<unsigned char*>{});
        image_publishers.emplace_back(std::move(node.advertise<sensor_msgs::Image>(tmp[0], 1)));
        camerainfo_publishers.emplace_back(std::move(node.advertise<sensor_msgs::CameraInfo>(tmp[0]+"/camera_info", 1)));

        ROS_INFO("Topic: %s", tmp[0].c_str());

        for (size_t i = 1; i < tmp.size(); ++i) {
          ROS_INFO("  %s", tmp[i].c_str());

          const std::string full_path{tmp[i][0] == '/'
                                      ? tmp[i]
                                      : pkg_path + tmp[i]};
          images_filepaths.back().push_back(full_path);
          CImg<unsigned char> img(full_path.c_str());
          W = img.width();
          H = img.height();
          unsigned char* raw_img = new unsigned char[W*H*3];
          size_t idx{0};
          switch (img.spectrum()) {
            case 1: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 0);
              }
              break;
            }
            case 3: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 1);
                raw_img[idx++] = img(x, y, 2);
              }
              break;
            }
            case 4: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 1);
                raw_img[idx++] = img(x, y, 2);
              }
              break;
            }
            default: {
              throw std::runtime_error("Weird image format");
            }
          }
          image_sets.back().push_back(raw_img);
        }
      }
    }
    tmp.clear();
    if (ros::param::get("~topic_2", tmp)) {
      if (tmp.size() > 1) {
        topics.push_back(tmp[0]);
        images_filepaths.push_back(std::vector<std::string>{});
        image_sets.push_back(std::vector<unsigned char*>{});
        image_publishers.emplace_back(std::move(node.advertise<sensor_msgs::Image>(tmp[0], 1)));
        camerainfo_publishers.emplace_back(std::move(node.advertise<sensor_msgs::CameraInfo>(tmp[0]+"/camera_info", 1)));

        ROS_INFO("Topic: %s", tmp[0].c_str());
        for (size_t i = 1; i < tmp.size(); ++i) {
          ROS_INFO("  %s", tmp[i].c_str());

          const std::string full_path{tmp[i][0] == '/'
                                      ? tmp[i]
                                      : pkg_path + tmp[i]};
          images_filepaths.back().push_back(full_path);
          CImg<unsigned char> img(full_path.c_str());
          W = img.width();
          H = img.height();
          unsigned char* raw_img = new unsigned char[W*H*3];
          size_t idx{0};
          switch (img.spectrum()) {
            case 1: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 0);
              }
              break;
            }
            case 3: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 1);
                raw_img[idx++] = img(x, y, 2);
              }
              break;
            }
            case 4: {
              cimg_forXY(img, x, y) {
                raw_img[idx++] = img(x, y, 0);
                raw_img[idx++] = img(x, y, 1);
                raw_img[idx++] = img(x, y, 2);
              }
              break;
            }
            default: {
              throw std::runtime_error("Weird image format");
            }
          }
          image_sets.back().push_back(raw_img);
        }
      }
    }
  }

  float fps{1.0f};
  ros::param::get("~fps", fps);
  ROS_INFO("Publishing at %f frames per second", fps);

  int repetitions{0};
  ros::param::get("~repetitions", repetitions);
  const bool forever{repetitions == 0};
  if (forever) {
    ROS_INFO("Publishing image sets indefinitely (infinite repetitions)");
  } else {
    if (repetitions == 1) {
      ROS_INFO("Publishing image sets one single time (no repetitions)");
    } else {
      ROS_INFO("Publishing image sets %d times", repetitions);
    }
  }

  float K[9] = {0, 0, 0,
                0, 0, 0,
                0, 0, 0};
  {
    std::vector<float> tmp;
    if (ros::param::get("~K_camera_intrinsics_matrix", tmp)) {
      if (tmp.size() == 9) {
        for (size_t i = 0; i < 9; ++i) {
          K[i] = tmp[i];
        }
        ROS_INFO("Using K intrinsics matrix:\n  %f %f %f\n  %f %f %f\n  %f %f %f",
                 K[0], K[1], K[2],
                 K[3], K[4], K[5],
                 K[6], K[7], K[8]);
      }
    }
  }

  /// Loop
  Pacemaker::Pacemaker publish_clock(fps);
  int seq_id{0};
  while (ros::ok()) {
    if (publish_clock.IsDue()) {
      static int img_idx{0};
      ros::Time now = ros::Time::now();
      for (size_t pub_idx = 0; pub_idx < image_publishers.size(); ++pub_idx) {
        send_image(image_sets[pub_idx][img_idx], W, H, 
                   image_publishers[pub_idx], now, seq_id);
        send_caminfo(W, H, K,
                     camerainfo_publishers[pub_idx], now, seq_id);
        ROS_INFO("%04d: %s --> %s", 
                 img_idx,
                 images_filepaths[pub_idx][img_idx].c_str(),
                 topics[pub_idx].c_str());
      }
      ros::spinOnce();
      img_idx = (img_idx+1)%(image_sets[0].size());
      if (img_idx == 0) {
        ROS_INFO("-- next repetition --");
        if (not forever) {
          --repetitions;
          if (repetitions == 0)
            break;
        }
      }
      ++seq_id;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /// Tidy up
  ROS_INFO("Exiting...");
  ros::shutdown();

  /// Bye!
  return EXIT_SUCCESS;
}

