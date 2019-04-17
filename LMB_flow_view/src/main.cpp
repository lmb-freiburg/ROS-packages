/**
 * Eddy Ilg, 2018 (ilg@cs.uni-freiburg.de)
 * Nikolaus Mayer, 2019
 *
 * This module provides flow visuzliation. It maps
 * the flow topic into an image topic (visualized flow).
 *
 * Input topic: /flownet
 * Outut topic: /flownet_viz
 *
 * Vizualization types (Parameter viz_type):
 *
 * "middlebury":
 *   Best used for computer screens, the center of the color circle is
 *   black (i.e. black indicates zero motion).
 *
 * "sintel":
 *   Best used for printing, the center of the color circle is
 *   white (i.e. white indicates zero motion).
 *
 *
 * Vizualization scale (Parameter viz_scale):
 *
 * The scale parameter is needed to scale the flows of your application
 * for apropriate visualization. Its value corresponds to the flow magnitude
 * at maximum saturation (choose appropriately for small or large flows).
 *
 */

/// Stdlib
#include <vector>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

/// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

/// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>

/// Flow visualization routines
#include "flow_viz.cpp"

/// Visualizaion type and scale
enum FlowVizType { VIZ_SINTEL, VIZ_MIDDLEBURY };

FlowVizType viz_type = VIZ_MIDDLEBURY;
float viz_scale = 10;

ros::Publisher publisher__visualized;

void input_callback(const sensor_msgs::ImageConstPtr& img_msg)
{
    ROS_INFO("Callback");

    /// Get images from messages
    int W = img_msg->width;
    int H = img_msg->height;

    /// Create output message
    sensor_msgs::Image flow_viz_image;
    flow_viz_image.header.stamp = img_msg->header.stamp;
    flow_viz_image.height = H;
    flow_viz_image.width = W;
    flow_viz_image.encoding = "rgb8";
    flow_viz_image.step = W*3;
    flow_viz_image.data.resize(W*H*3);

    /// Do conversion
    if(viz_type == VIZ_MIDDLEBURY) flow_viz_middlebury((float*)&img_msg->data[0], &flow_viz_image.data[0], W, H, viz_scale);
    else                           flow_viz_sintel(    (float*)&img_msg->data[0], &flow_viz_image.data[0], W, H, viz_scale);

    /// Publish
    publisher__visualized.publish(flow_viz_image);
}

int main(int argc, char** argv)
{
    /// Initialization
    ros::init(argc, argv, "flownet_vis");
    ROS_INFO("Node initialized");

    std::string viz_type_param;
    if (ros::param::get("~viz_type", viz_type_param)) {
        ROS_INFO("Setting vizualization type to: %s", viz_type_param.c_str());
        if(viz_type_param == "middlebury") viz_type = VIZ_MIDDLEBURY;
        else viz_type = VIZ_SINTEL;
    }

    if (ros::param::get("~viz_scale", viz_scale)) {
        ROS_INFO("Setting vizualization scale to: %s", viz_scale);
    }

    ros::NodeHandle node;

    /// Advertise visualized topic
    publisher__visualized = node.advertise<sensor_msgs::Image>("/flownet_vis", 1);

    /// Input topics
    ROS_INFO("Subscribing to flow image stream");
    ros::Subscriber image_subscriber = node.subscribe(node.resolveName("flownet"), 1, input_callback);

    ros::spin();

    /// Tidy up
    ROS_INFO("Exiting...");
    ros::shutdown();

    /// Bye!
    return EXIT_SUCCESS;
}

