/// Nikolaus Mayer, 2019

/// System/STL
#include <condition_variable>
#include <chrono>
#include <cmath>
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
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PolygonStamped.h>
#include <tf/transform_listener.h>
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
/// Eigen3
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/SVD>
#include <Eigen/LU>  // Matrix determinant
#include <Eigen/Eigenvalues>

typedef pcl::PointXYZ Point_3;
typedef pcl::PointCloud<Point_3> PCL_3;
typedef pcl::PointXYZRGB Point_6;
typedef pcl::PointCloud<Point_6> PCL_6;

tf::TransformListener* tf_ptr;

/// Node parameters
std::string mode{"adjacent-frames"};
bool invert_fx{false};
bool invert_fy{false};
bool invert_cx{false};
bool invert_cy{false};
bool camera_tf{false};
float scene_motion_threshold{0.05f};
float ignore_beyond{1.5f};

/// Camera intrinsics matrix and its inverse
Eigen::Matrix3f K_left, iK_left;

/// Publishers for this node's topics
ros::Publisher pcl_pub, pcl_pub_2, pcl_pub_3, tracked_points_3d;
ros::Publisher arrows_pub;
ros::Publisher sceneflow_img_vis;
ros::Publisher warped_image_pub;
ros::Publisher publisher__camera_frustum;

std::vector<float> points_to_track;
std::vector<float> points_to_track__tracked;

/// Placeholders for inputs and outputs
stereo_msgs::DisparityImage previous_disparity, 
                            current_disparity;
sensor_msgs::Image previous_left,
                   previous_right,
                   current_left,
                   current_right,
                   left_flow;

/**
 * Access an integer pixel in a DisparityImage message
 */
float disp_lookup(const stereo_msgs::DisparityImage& disp, int x, int y)
{
  if (x < 0 or x >= disp.image.width or
      y < 0 or y >= disp.image.height)
    return 0;
  const size_t pos{y * disp.image.width + x};
  return reinterpret_cast<float const*>(disp.image.data.data())[pos];
}

/**
 * Access a non-integer pixel in a DisparityImage message, using bilinear
 * interpolation
 */
float bilinear_disp_lookup(const stereo_msgs::DisparityImage& disp, float x, float y)
{
  if (x < 0 or x >= disp.image.width or
      y < 0 or y >= disp.image.height)
    return 0;

  int x0{(int)x};
  int y0{(int)y};
  float xf{x-x0};
  float yf{y-y0};
  int x1 = (xf < 0.001f) ? x0 : x0+1;
  int y1 = (yf < 0.001f) ? y0 : y0+1;
  return (1-xf)*(1-yf)*disp_lookup(disp,x0,y0) +
            xf *(1-yf)*disp_lookup(disp,x1,y0) +
         (1-xf)*   yf *disp_lookup(disp,x0,y1) +
            xf *   yf *disp_lookup(disp,x1,y1);
}

/**
 * Access an integer pixel in the left_flow map
 */
float flow_lookup(int x, int y, int z)
{
  if (x < 0 or x >= left_flow.width or
      y < 0 or y >= left_flow.height)
    return 0;
  const size_t pos{y * left_flow.width + x};
  return reinterpret_cast<float*>(left_flow.data.data())[2*pos+z];
}

/**
 * Access a non-integer pixel in the left_flow, using bilinear
 * interpolation
 */
float bilinear_flow_lookup(float x, float y, int z)
{
  if (x < 0 or x >= left_flow.width or
      y < 0 or y >= left_flow.height)
    return 0;

  int x0{(int)x};
  int y0{(int)y};
  float xf{x-x0};
  float yf{y-y0};
  int x1 = (xf < 0.001f) ? x0 : x0+1;
  int y1 = (yf < 0.001f) ? y0 : y0+1;
  return (1-xf)*(1-yf)*flow_lookup(x0,y0,z) +
            xf *(1-yf)*flow_lookup(x1,y0,z) +
         (1-xf)*   yf *flow_lookup(x0,y1,z) +
            xf *   yf *flow_lookup(x1,y1,z);
}

/**
 * Access a color value at an integer pixel position in an image message
 */
unsigned char image_lookup(const sensor_msgs::Image& image, int x, int y, int z)
{
  if (x < 0 or x >= image.width or
      y < 0 or y >= image.height)
    return 0;
  const size_t pos{y * image.width + x};
  return image.data[3*pos+z];
}

/**
 * Access a color value at a noninteger pixel position in an image message,
 * using bilinear interpolation
 */
unsigned char bilinear_image_lookup(const sensor_msgs::Image& image, float x, float y, int z)
{
  if (x < 0 or x >= image.width or
      y < 0 or y >= image.height)
    return 0;

  int x0{(int)x};
  int y0{(int)y};
  float xf{x-x0};
  float yf{y-y0};
  int x1 = (xf < 0.001f) ? x0 : x0+1;
  int y1 = (yf < 0.001f) ? y0 : y0+1;
  return (1-xf)*(1-yf)*image_lookup(image,x0,y0,z) +
            xf *(1-yf)*image_lookup(image,x1,y0,z) +
         (1-xf)*   yf *image_lookup(image,x0,y1,z) +
            xf *   yf *image_lookup(image,x1,y1,z);
}


/**
 * Callback for input/output data from main sceneflow node
 */
void callback(const sensor_msgs::ImageConstPtr& img_msg_l,
              const sensor_msgs::ImageConstPtr& img_msg_r,
              const sensor_msgs::CameraInfoConstPtr& caminfo_msg_l,
              const stereo_msgs::DisparityImageConstPtr& disp_msg,
              const sensor_msgs::ImageConstPtr& flow_msg)
{
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
    iK_left << K_left.inverse();
  }

  {
    previous_left  = current_left;
    previous_right = current_right;
    previous_disparity = current_disparity;
  }

  current_left      = *img_msg_l;
  current_right     = *img_msg_r;
  current_disparity = *disp_msg;
  left_flow         = *flow_msg;
  if (previous_left.width == 0) {
    previous_left  = current_left;
    previous_right = current_right;
    previous_disparity = current_disparity;
    return;
  }

  /// Get camera pose
  static tf::StampedTransform current_pose, previous_pose;
  if (camera_tf) {
    previous_pose = current_pose;
    
    try {
      tf_ptr->waitForTransform("camera", "world", 
                              current_left.header.stamp, ros::Duration(3.0));
      tf_ptr->lookupTransform("camera", "world", 
                              current_left.header.stamp, current_pose);
    } catch (const std::exception& e) {
      ROS_INFO("NO TF RAW: %s", e.what());
      return;
    }

    current_pose.setData(current_pose.inverse());

  } else {
    previous_pose.setIdentity();
    current_pose.setIdentity();
  }





  /// Transform to a nicely viewable arbitrary orientation
  tf::StampedTransform tilt;
  {
    tilt.setIdentity();
    //tf::Matrix3x3 tmp;
    //tmp.setEulerYPR(0, 0, -3.14/2);
    //tilt.setBasis(tmp);
  }


  /// Visualize camera pose
  {
    PCL_6::Ptr frustum_cloud(new PCL_6());
    auto add_point = [&iK_left,&frustum_cloud](float x,float y,float z)->void {
      Eigen::Vector3f point{x, y, 1.f};
      point = (iK_left*point)*(z/point(2));
      Point_6 pt;
      pt.x = point(0);
      pt.y = point(1);
      pt.z = point(2);
      pt.r = 0;
      pt.g = 255;
      pt.b = 0;
      frustum_cloud->points.push_back(pt);
    };

    geometry_msgs::PolygonStamped frustum_msg;
    frustum_msg.header.stamp    = current_left.header.stamp;
    frustum_msg.header.frame_id = "root";
    const float W = current_left.width;
    const float H = current_left.height;
    const float camera_frustum_size{0.2f};
    add_point(0,0,-camera_frustum_size);
    add_point(W,0,-camera_frustum_size);
    add_point(0,0,0);
    add_point(0,0,-camera_frustum_size);
    add_point(0,H,-camera_frustum_size);
    add_point(W,H,-camera_frustum_size);
    add_point(W,0,-camera_frustum_size);
    add_point(0,0,0);
    add_point(W,H,-camera_frustum_size);
    add_point(0,0,0);
    add_point(0,H,-camera_frustum_size);

    /// "Notch"
    add_point(0,0,-camera_frustum_size);
    add_point(0.4*W,0,-camera_frustum_size);
    add_point(0.5*W,-0.1*H,-camera_frustum_size);
    add_point(0.6*W,0,-camera_frustum_size);

    pcl_ros::transformPointCloud(*frustum_cloud, 
                                 *frustum_cloud, 
                                 current_pose);
    pcl_ros::transformPointCloud(*frustum_cloud, 
                                 *frustum_cloud, 
                                 tilt);

    for (const auto& point: frustum_cloud->points) {
      geometry_msgs::Point32 p;
      p.x = point.x; p.y = point.y; p.z = point.z;
      frustum_msg.polygon.points.push_back(p);
    }
    publisher__camera_frustum.publish(frustum_msg);
  }


  /// Track points
  {
    const unsigned int W{current_left.width};
    const unsigned int H{current_left.height};

    static Eigen::Vector3f startcam;

    {
      auto make_3d_point = [&](int x, int y, Eigen::Vector3f& point) -> bool {
        point = Eigen::Vector3f(0,0,0);
        float v{disp_lookup(current_disparity, x, y)};
        if (std::abs(v) == 0.f or std::abs(v) == 0.f)
          return false;
        v = -0.0496*K_left(0,0)/v;
        if (std::abs(v) > ignore_beyond)
          return false;
        const float _x{invert_fx ? (W-1.f-x) : x};
        const float _y{invert_fy ? (H-1.f-y) : y};
        //Eigen::Vector3f point{_x, _y, 1.f};
        point = Eigen::Vector3f(_x, _y, 1.f);
        point = (iK_left*point)*(v/point(2));
        return true;
      };

      static int framecounter{0};
      static std::vector<std::vector<std::vector<float>>> tracked_points_xy;
      static std::vector<std::vector<Eigen::Vector3f>> tracked_points;
      static std::vector<bool> tracked_points_ok;
      if (framecounter == 0) {
        startcam = Eigen::Vector3f{current_pose.getOrigin().x(),
                                   current_pose.getOrigin().y(),
                                   current_pose.getOrigin().z()};
        PCL_3 tmp_pcl;
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            Eigen::Vector3f point;
            if (make_3d_point(x, y, point)) {
              tracked_points_xy.push_back({{x,y}});
              tracked_points_ok.push_back(true);
            } else {
              tracked_points_xy.push_back({{}});
              point = Eigen::Vector3f(0,0,0);
              tracked_points_ok.push_back(false);
            }
            tracked_points.push_back({});

            Point_3 pt;
            pt.x = point(0);
            pt.y = point(1);
            pt.z = point(2);
            tmp_pcl.points.push_back(pt);
          }
        }
        pcl_ros::transformPointCloud(tmp_pcl, tmp_pcl, current_pose);
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const size_t linear_idx{y*W+x};
            if (tracked_points_ok[linear_idx]) {
              Eigen::Vector3f point(tmp_pcl.points[linear_idx].x,
                                    tmp_pcl.points[linear_idx].y,
                                    tmp_pcl.points[linear_idx].z);
              tracked_points[linear_idx].push_back(point);
            }
          }
        }
      } else {
        PCL_3 tmp_pcl;
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const size_t linear_idx{y*W+x};
            //if (not tracked_points_ok[linear_idx])
            //  continue;
            Eigen::Vector3f point(0,0,0);
            //const bool ok = make_3d_point(x, y, point);
            //if (not ok)
            //  tracked_points_ok[linear_idx] = false;
            if (tracked_points_ok[linear_idx]) {
              float px = tracked_points_xy[linear_idx].back()[0];
              float py = tracked_points_xy[linear_idx].back()[1];
              const float xf = bilinear_flow_lookup(px, py, 0);
              const float yf = bilinear_flow_lookup(px, py, 1);
              px = std::max(0.f, std::min(px+xf, W-1.f));
              py = std::max(0.f, std::min(py+yf, H-1.f));
              if (make_3d_point(px, py, point)) {
                tracked_points_xy[linear_idx].push_back({px, py});
                //tracked_points[linear_idx].push_back(point);
              } else {
                tracked_points_ok[linear_idx] = false;
                point = Eigen::Vector3f(0,0,0);
              }
            }

            Point_3 pt;
            pt.x = point(0);
            pt.y = point(1);
            pt.z = point(2);
            tmp_pcl.points.push_back(pt);
          }
        }
        pcl_ros::transformPointCloud(tmp_pcl, tmp_pcl, current_pose);
        size_t ok_count{0};
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            const size_t linear_idx{y*W+x};
            if (tracked_points_ok[linear_idx]) {
              ++ok_count;
              Eigen::Vector3f point(tmp_pcl.points[linear_idx].x,
                                    tmp_pcl.points[linear_idx].y,
                                    tmp_pcl.points[linear_idx].z);
              tracked_points[linear_idx].push_back(point);
            }
          }
        }
        ROS_INFO("OK count: %d of %d points ok", ok_count, W*H);

        /// Evaluation / debugging output
        /*if (framecounter > 1) {
          char filename[16];
          std::snprintf(filename, 16, "logfile-%03d.csv", framecounter);
          ROS_INFO(" -> %s", filename);
          std::ofstream logfile(filename);
          if (logfile.bad() or not logfile.is_open())
            throw std::runtime_error("could not open logfile for writing");
          for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
              const size_t linear_idx{y*W+x};
              if (not tracked_points_ok[linear_idx])
                continue;
              const std::vector<Eigen::Vector3f>& ptlist{tracked_points[linear_idx]};
              if (ptlist.size() < 2)
                continue;

              const float initial_distance_from_camera = (ptlist[0]-startcam).norm();
              const float final_position_error = (ptlist.back()-ptlist[0]).norm();
              logfile << initial_distance_from_camera << ", "
                      << final_position_error << std::endl;
            }
          }
          logfile.close();
        }*/
      }
      ++framecounter;
    }

    //static PCL_6 tracked_points_pcl;
    //tracked_points_pcl.points.clear();
    for (size_t i = 0; i < points_to_track__tracked.size(); i+=2) {
      float x{points_to_track__tracked[i]};
      float y{points_to_track__tracked[i+1]};
      if (mode == "to-first-frame") {
        x = points_to_track[i];
        y = points_to_track[i+1];
      }
      size_t pos{(int)y * left_flow.width + (int)x};
      float tmp{x};
      x += bilinear_flow_lookup(x, y, 0);
      y += bilinear_flow_lookup(tmp, y, 1);
      x = std::max(0.f, std::min(x, W-1.f));
      y = std::max(0.f, std::min(y, H-1.f));
      points_to_track__tracked[i]   = x;
      points_to_track__tracked[i+1] = y;

      {
        float v{disp_lookup(current_disparity, x, y)};
        if (std::abs(v) == 0.f or std::abs(v) == 0.f)
          continue;

        v = -0.0496*K_left(0,0)/v;
        if (std::abs(v) > ignore_beyond)
          continue;
        
        const float _x{invert_fx ? (W-1.f-x) : x};
        const float _y{invert_fy ? (H-1.f-y) : y};
        Eigen::Vector3f point{_x, _y, 1.f};
        point = (iK_left*point)*(v/point(2));

        //Point_6 pt;
        //pt.x = point(0);
        //pt.y = point(1);
        //pt.z = point(2);
        //pt.r = 255;
        //pt.g = 0;
        //pt.b = 0;

        //tracked_points_pcl.points.push_back(pt);
      }
    }
    //tracked_points_pcl.header.frame_id = "root";
    //pcl_conversions::toPCL(current_left.header.stamp, 
    //                       tracked_points_pcl.header.stamp);

    //pcl_ros::transformPointCloud(tracked_points_pcl, tracked_points_pcl, current_pose);
    //pcl_ros::transformPointCloud(tracked_points_pcl, tracked_points_pcl, tilt);
    //tracked_points_3d.publish(tracked_points_pcl);
  }
  

  {
    cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(current_left, 
                             sensor_msgs::image_encodings::TYPE_8UC3);
    const unsigned int W{current_left.width};
    const unsigned int H{current_left.height};
    sensor_msgs::Image vis_msg;
    vis_msg.header.stamp = current_left.header.stamp;
    vis_msg.height = H;
    vis_msg.width = W;
    vis_msg.encoding = "rgb8";
    vis_msg.step = W*3;
    vis_msg.data = std::vector<uint8_t>(W*H*3);
    for (unsigned int y = 0; y < H; ++y) {
      for (unsigned int x = 0; x < W; ++x) {
        vis_msg.data.data()[3*(y*W+x)+0] = cv_img->image.at<cv::Vec3b>(y,x)[2];
        vis_msg.data.data()[3*(y*W+x)+1] = cv_img->image.at<cv::Vec3b>(y,x)[1];
        vis_msg.data.data()[3*(y*W+x)+2] = cv_img->image.at<cv::Vec3b>(y,x)[0];
      }
    }
    for (size_t i = 0; i < points_to_track.size(); i+=2) {
      float& x{points_to_track__tracked[i]};
      float& y{points_to_track__tracked[i+1]};
      for (int dx = -10; dx <= 10; ++dx) {
        for (int dy = -10; dy <= 10; ++dy) {
          if (x+dx < 0 or x+dx >= W or y+dy < 0 or y+dy >= H)
            continue;
          const float dist{std::sqrt(dx*dx+dy*dy)};
          if (dist < 5) {
            continue;
          } else if (dist <= 7) {
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+0] = 255;
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+1] = 255;
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+2] = 255;
          } else if (dist <= 10) {
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+0] = 0;
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+1] = 0;
            vis_msg.data.data()[3*(((int)y+dy)*W+((int)x+dx))+2] = 0;
          }
        }
      }
    }
    sceneflow_img_vis.publish(vis_msg);



    visualization_msgs::MarkerArray trajectory_msg;

    PCL_6 pointcloud_prev, pointcloud_curr, pointcloud_class, pointcloud_corr;
    int marker_id{0};
    for (unsigned int y = 0; y < H; y+=2) {
      for (unsigned int x = 0; x < W; x+=2) {
        float v{disp_lookup(previous_disparity, x, y)};
        float v2{disp_lookup(current_disparity, x, y)};
        if (std::abs(v) == 0.f or std::abs(v2) == 0.f)
          continue;

        const float xf{flow_lookup(x,y,0)};
        const float yf{flow_lookup(x,y,1)};
        if (std::isnan(xf) or std::isnan(yf))
          continue;

        /// We also skip "far away" points; they are likely low-quality and 
        /// we do not use them anyway
        v = -0.0496*K_left(0,0)/v;
        if (std::abs(v) > ignore_beyond)
          continue;
        
        const float _x{invert_fx ? (W-1.f-x) : x};
        const float _y{invert_fy ? (H-1.f-y) : y};
        Eigen::Vector3f point{_x, _y, 1.f};
        point = (iK_left*point)*(v/point(2));

        Point_6 pt;
        pt.x = point(0);
        pt.y = point(1);
        pt.z = point(2);
        pt.r = bilinear_image_lookup(previous_left, x, y, 2);
        pt.g = bilinear_image_lookup(previous_left, x, y, 1);
        pt.b = bilinear_image_lookup(previous_left, x, y, 0);
        //unsigned char gray = 0.3*bilinear_image_lookup(previous_left, _x, _y, 2)+
        //                     0.3*bilinear_image_lookup(previous_left, _x, _y, 1)+
        //                     0.3*bilinear_image_lookup(previous_left, _x, _y, 0);
        //pt.r = gray;
        //pt.g = gray;
        //pt.b = gray;

        if (pt.r == 0 and pt.g == 0 and pt.b == 0)
          continue;
      /*}
    }
    for (unsigned int y = 0; y < H; y+=1) {
      for (unsigned int x = 0; x < W; x+=1) {*/
        //float v2{disp_lookup(current_disparity, x, y)};
        //float v = bilinear_disp_lookup(current_disparity,
        //                               x+xf,
        //                               y+yf);
        /// We also skip "far away" points; they are likely low-quality and 
        /// we do not use them anyway
        v2 = -0.0496*K_left(0,0)/v2;
        if (std::abs(v2) > ignore_beyond)
          continue;
        
        //const float _x{x};
        //const float _y{y};
        Eigen::Vector3f point2{_x, _y, 1.f};
        //Eigen::Vector3f point2{_x+xf, _y+yf, 1.f};
        point2 = (iK_left*point2)*(v2/point2(2));

        Point_6 pt2;
        pt2.x = point2(0);
        pt2.y = point2(1);
        pt2.z = point2(2);
        pt2.r = bilinear_image_lookup(current_left, _x, _y, 2);
        pt2.g = bilinear_image_lookup(current_left, _x, _y, 1);
        pt2.b = bilinear_image_lookup(current_left, _x, _y, 0);
        //pt2.r = bilinear_image_lookup(current_left, _x+xf, _y+yf, 2);
        //pt2.g = bilinear_image_lookup(current_left, _x+xf, _y+yf, 1);
        //pt2.b = bilinear_image_lookup(current_left, _x+xf, _y+yf, 0);

        if (pt2.r == 0 and pt2.g == 0 and pt2.b == 0)
          continue;


        pointcloud_prev.points.push_back(pt);
        pointcloud_curr.points.push_back(pt2);

        {
          float vc = bilinear_disp_lookup(current_disparity,
                                          x+xf,
                                          y+yf);
          vc = 0.05*K_left(0,0)/vc;
          Eigen::Vector3f pc{_x+xf, _y+yf, 1.f};
          pc = (iK_left*pc)*(vc/pc(2));
          Point_6 ptc;
          ptc.x = pc(0);
          ptc.y = pc(1);
          ptc.z = pc(2);
          pointcloud_corr.points.push_back(ptc);
        }
      }
    }





    //  }
    //}
    pcl_ros::transformPointCloud(pointcloud_prev, pointcloud_prev, previous_pose);
    pcl_ros::transformPointCloud(pointcloud_prev, pointcloud_prev, tilt);
    pointcloud_prev.header.frame_id = "root";
    pcl_conversions::toPCL(current_left.header.stamp, pointcloud_prev.header.stamp);
    pcl_pub.publish(pointcloud_prev);

    pcl_ros::transformPointCloud(pointcloud_curr, pointcloud_curr, current_pose);
    pcl_ros::transformPointCloud(pointcloud_curr, pointcloud_curr, tilt);
    pointcloud_curr.header.frame_id = "root";
    pcl_conversions::toPCL(current_left.header.stamp, pointcloud_curr.header.stamp);
    pcl_pub_2.publish(pointcloud_curr);

    pcl_ros::transformPointCloud(pointcloud_corr, pointcloud_corr, current_pose);
    pcl_ros::transformPointCloud(pointcloud_corr, pointcloud_corr, tilt);

    {
      int px{0};
      int py{0};
      for (size_t pt_idx = 0; pt_idx < pointcloud_prev.points.size(); ++pt_idx) {
        const Point_6 pt_prev{pointcloud_prev.points[pt_idx]};
        const Point_6 pt_curr{pointcloud_corr.points[pt_idx]};
        Point_6 cls;
        cls.x = pt_prev.x;
        cls.y = pt_prev.y;
        cls.z = pt_prev.z;
        const float dist{std::sqrt((pt_prev.x-pt_curr.x)*(pt_prev.x-pt_curr.x) +
                                   (pt_prev.y-pt_curr.y)*(pt_prev.y-pt_curr.y) +
                                   (pt_prev.z-pt_curr.z)*(pt_prev.z-pt_curr.z))};
        if (dist < scene_motion_threshold) {
          cls.r = 0;
          cls.g = 0;
          cls.b = 255;
        } else {
          cls.r = 255;
          cls.g = 0;
          cls.b = 0;
        }
        pointcloud_class.points.push_back(cls);

        if (px%40==0 and py%30==0) {
          visualization_msgs::Marker marker;
          marker.header.stamp    = current_left.header.stamp;
          marker.header.frame_id = "root";
          marker.ns              = "LMB__cam_path";
          marker.id              = marker_id++;
          marker.type            = 0;  // ARROW=0
          marker.action          = 0;  // ADD/MODIFY=0
          marker.scale.x         = 0.006f;
          marker.scale.y         = 0.01f;
          marker.scale.z         = 0.01f;
          marker.color.r         = 1.f;
          marker.color.g         = 0.f;
          marker.color.b         = 0.f;
          if (dist > 0.2 or dist < 0.02)
            marker.color.a         = 0.f;
          else 
            marker.color.a         = 1.f;
          {
            geometry_msgs::Point p0, p1;
            p0.x = pt_prev.x;
            p0.y = pt_prev.y;
            p0.z = pt_prev.z;
            p1.x = pt_curr.x;
            p1.y = pt_curr.y;
            p1.z = pt_curr.z;
            marker.points.push_back(p0);
            marker.points.push_back(p1);
          }
          trajectory_msg.markers.push_back(marker);
          arrows_pub.publish(trajectory_msg);
        }

        ++px;
        if (px == current_left.width) {
          px = 0;
          ++py;
        }
      }
    }

    pointcloud_class.header.frame_id = "root";
    pcl_conversions::toPCL(current_left.header.stamp, pointcloud_class.header.stamp);
    pcl_pub_3.publish(pointcloud_class);

  }
  {
    cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(previous_left, 
                             sensor_msgs::image_encodings::TYPE_8UC3);
    const unsigned int W{previous_left.width};
    const unsigned int H{previous_left.height};
    sensor_msgs::Image vis_msg;
    vis_msg.header.stamp = current_left.header.stamp;
    vis_msg.height = H;
    vis_msg.width = W;
    vis_msg.encoding = "rgb8";
    vis_msg.step = W*3;
    vis_msg.data = std::vector<uint8_t>(W*H*3);
    for (unsigned int y = 0; y < H; ++y) {
      for (unsigned int x = 0; x < W; ++x) {
        const float fx{bilinear_flow_lookup(x,y,0)};
        const float fy{bilinear_flow_lookup(x,y,1)};
        if (std::abs(fx) < 0.1 and std::abs(fy) < 0.1) {
          vis_msg.data.data()[3*(y*W+x)+0] = image_lookup(previous_left,x,y,2);
          vis_msg.data.data()[3*(y*W+x)+1] = image_lookup(previous_left,x,y,1);
          vis_msg.data.data()[3*(y*W+x)+2] = image_lookup(previous_left,x,y,0);
        } else {
          vis_msg.data.data()[3*(y*W+x)+0] = bilinear_image_lookup(current_left,x+fx,y+fy,2);
          vis_msg.data.data()[3*(y*W+x)+1] = bilinear_image_lookup(current_left,x+fx,y+fy,1);
          vis_msg.data.data()[3*(y*W+x)+2] = bilinear_image_lookup(current_left,x+fx,y+fy,0);
        }
      }
    }
    warped_image_pub.publish(vis_msg);
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "sceneflowvis");
  ros::NodeHandle node;

  /// Parameters
  ros::param::param<std::string>("~mode", mode, "adjacent-frames");
  ros::param::param<bool>("~invert_fx", invert_fx, false);
  ros::param::param<bool>("~invert_fy", invert_fy, false);
  ros::param::param<bool>("~invert_cx", invert_cx, false);
  ros::param::param<bool>("~invert_cy", invert_cy, false);
  ros::param::param<bool>("~camera_tf", camera_tf, false);
  ros::param::param<float>("~scene_motion_threshold", 
                             scene_motion_threshold, 0.05f);
  ros::param::param<float>("~ignore_beyond", ignore_beyond, 1.5f);
  
  pcl_pub                   = node.advertise<PCL_6>(
                                    "LMB__sceneflow_pcl", 10);
  pcl_pub_2                 = node.advertise<PCL_6>(
                                    "LMB__sceneflow_pcl_2", 10);
  pcl_pub_3                 = node.advertise<PCL_6>(
                                    "LMB__sceneflow_pcl_3", 10);
  tracked_points_3d         = node.advertise<PCL_6>(
                                    "LMB__sceneflow_trackedpoints", 10);
  arrows_pub                = node.advertise<visualization_msgs::MarkerArray>(
                                    "LMB__sceneflow_arrows", 10);
  sceneflow_img_vis         = node.advertise<sensor_msgs::Image>(
                                    "LMB__sceneflow_vis", 1);
  warped_image_pub          = node.advertise<sensor_msgs::Image>(
                                    "LMB__sceneflow_warped", 1);
  publisher__camera_frustum = node.advertise<geometry_msgs::PolygonStamped>(
                                    "LMB__cam_view", 1);


  ros::param::get("~points_to_track", points_to_track);
  //for (int y = 0; y < 480-50; y+=15) {
  //  for (int x = 20; x < 752-30; x+=15) {
  //    points_to_track.push_back(x);
  //    points_to_track.push_back(y);
  //  }
  //}
  points_to_track__tracked = points_to_track;

  tf_ptr = new tf::TransformListener(node, ros::Duration(30));
  
  /// Subscribe to rectified images (INPUT callback)
  ROS_INFO("Registering sync'ed image streams");
  ros::TransportHints my_hints{ros::TransportHints().tcpNoDelay(true)};
  std::string topic_cam_l = node.resolveName("LMB__sceneflow_left");
  std::string topic_cam_r = node.resolveName("LMB__sceneflow_right");
  std::string topic_caminfo_l = node.resolveName("LMB__sceneflow_left/camera_info");
  std::string topic_disp = node.resolveName("dispnet");
  std::string topic_flow = node.resolveName("flownet");
  message_filters::Subscriber<sensor_msgs::Image> cam_l_sub(node,
                                                  topic_cam_l, 1, my_hints);
  message_filters::Subscriber<sensor_msgs::Image> cam_r_sub(node,
                                                  topic_cam_r, 1, my_hints);
  message_filters::Subscriber<sensor_msgs::CameraInfo> caminfo_l_sub(node, 
                                                  topic_caminfo_l, 1, my_hints);
  message_filters::Subscriber<stereo_msgs::DisparityImage> disp_sub(node, 
                                                  topic_disp, 1, my_hints);
  message_filters::Subscriber<sensor_msgs::Image> flow_sub(node, 
                                                  topic_flow, 1, my_hints);
  message_filters::TimeSynchronizer<sensor_msgs::Image, 
                                    sensor_msgs::Image,
                                    sensor_msgs::CameraInfo,
                                    stereo_msgs::DisparityImage,
                                    sensor_msgs::Image> sync(cam_l_sub, 
                                                             cam_r_sub, 
                                                             caminfo_l_sub, 
                                                             disp_sub,
                                                             flow_sub,
                                                             1);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5));

  ros::spin();


  /// Bye!
  return EXIT_SUCCESS;
}

