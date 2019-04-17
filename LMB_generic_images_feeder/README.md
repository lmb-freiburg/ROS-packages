
LMB Generic Images Feeder
==========================

Description
-----------

LMB's Generic Images Feeder (AGIF) is a ROS node that repeatedly publishes
fixed sets of images.


Launchfile parameters
---------------------

- "topic_1": A list of strings. This parameter describes an images topic to be
             published. The first entry in the list is the name of the topic;
             all entries after that are the ordered list of image files that
             will be read and published. The image file paths are RELATIVE TO
             THE NODE'S FOLDER within the workspace.

- "topic_2": Same as "topic_1". Allows for multiple image topics to be published.
             The two topics will be synchronized, and both topics must be given
             the same number of image files. More topics ("topic_3" etc.) are
             an easy extension to the current implementation.

- "K_camera_intrinsics_matrix": A list of floats. Camera intrinsics matrix.
             Will be used to fill the published image topics' camera_info "K"
             field. Row-major notation ("left-to-right, top-to-bottom").

- "repetitions": An integer. Specifies how many times the image lists shall
             be published for the given image topics.
             If "repetitions" is 0 (zero), the topics are repeated forever.

- "fps": A float number. Specifies how many images per second shall be published
             in the image topics. Useful for throttling.



Example launchfile
------------------

This example launchfile will publish 2 topics ("/uvc_cam1_rect_mono" and
"/uvc_cam0_rect_mono") with 2 images each (all in the "data" subfolder of the
node). 2 Images will be published per second, and the repetitions will go on
forever.


```
<launch>
  <node pkg="LMB_generic_images_feeder"
        type="images_feeder"
        name="generic_images_feeder"
        output="screen"
  >
    <rosparam param="topic_1">
      [
        /uvc_cam1_rect_mono,
        /data/00100_L.png,
        /data/00101_L.png,
      ]
    </rosparam>
    <rosparam param="topic_2">
      [
        /uvc_cam0_rect_mono,
        /data/00100_R.png,
        /data/00101_R.png,
      ]
    </rosparam>

    <rosparam param="K_camera_intrinsics_matrix">
      [
        1050, 0, 1,
        0, 1050, 1,
        0, 0, 1
      ]
    </rosparam>

    <param name="repetitions" type="int" value="0"/>
    <param name="fps" type="double" value="2.0"/>
  </node>
</launch>
```


