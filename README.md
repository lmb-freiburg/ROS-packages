# ROS-packages
A collection of ROS packages for LMB software

# License
Non-commercial, for research purposes only.

# Contents
Most packages in this repository are neural networks based on the Caffe library. Use [our modified legacy Caffe code](https://github.com/lmb-freiburg/flownet2).
### Disparity
- `LMB_dispnet` runs DispNet disparity estimation networks -- mostly [the newer versions](https://lmb.informatik.uni-freiburg.de/Publications/2018/ISKB18/).
- `LMB_disparity_view` converts the outputs of `LMB_dispnet` into displayable images.
### Optical flow
- `LMB_flownet` and `LMB_flow_view` do the same thing as `LMB_dispnet` and `LMB_disparity_view` (see above), but for optical flow.
### Scene flow
- `LMB_sceneflow` runs `LMB_dispnet` and `LMB_flownet` nodes for scene flow (= dense 3D motion vectors) estimation.
- `LMB_sceneflow_vis` contains various visualizations for scene flow.
### Data input
- `LMB_generic_images_feeder` is a simple node that reads loose collections of image files and produces ROS image topics.


# Installation
- Get additional binary package with pretrained network weights from [LMB](https://lmb.informatik.uni-freiburg.de/data/GitHub/ROS-packages/ROS-packages-data-1.tar.gz)
- Build using catkin (tested with ROS Kinetic)


Author: Nikolaus Mayer, 2019

This repository does not accept pull requests.

The software in this repository is part of the Horizon2020 project [Trimbot2020](www.trimbot2020.org).

