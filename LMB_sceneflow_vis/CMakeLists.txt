cmake_minimum_required(VERSION 2.8.3)
project(LMB_sceneflow_vis)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge image_transport message_filters roscpp roslib std_msgs pcl_ros visualization_msgs
)

find_package( Threads REQUIRED )
find_package( PCL REQUIRED )
find_package( Eigen3 REQUIRED )

## Enable compiler warnings
set( CMAKE_CXX_FLAGS "-W -Wall -Wextra -std=c++11" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0" )
set( CMAKE_CXX_FLAGS_RELEASE "-O2" )

add_definitions(${PCL_DEFINITIONS})
add_definitions(${Eigen_DEFINITIONS})

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp roslib std_msgs visualization_msgs
)

include_directories(
  SYSTEM ${PCL_INCLUDE_DIRS}
  SYSTEM ${Eigen_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_executable(sceneflowvis src/main.cpp)

target_link_libraries(
  sceneflowvis
  ${PCL_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${catkin_LIBRARIES}
)

install(TARGETS sceneflowvis
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch                   
)   

