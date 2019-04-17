cmake_minimum_required(VERSION 2.8.3)
project(LMB_sceneflow)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge image_transport message_filters roscpp roslib std_msgs pcl_ros visualization_msgs
)

find_package( Threads REQUIRED )
find_package( PCL REQUIRED )

## Enable compiler warnings
set( CMAKE_CXX_FLAGS "-W -Wall -Wextra -std=c++11" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0" )
set( CMAKE_CXX_FLAGS_RELEASE "-O2" )

add_definitions(${PCL_DEFINITIONS})

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp roslib std_msgs visualization_msgs
)

include_directories(
  SYSTEM ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_executable(sceneflow src/main.cpp)

target_link_libraries(
  sceneflow
  ${PCL_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${catkin_LIBRARIES}
)

install(TARGETS sceneflow
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch                   
)   

