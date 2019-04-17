cmake_minimum_required(VERSION 2.8.3)
project(LMB_flow_view)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  roslib
  std_msgs
  stereo_msgs
)

find_package( Threads REQUIRED )

set( CMAKE_CXX_FLAGS "-W -Wall -Wextra -std=c++11" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3" )

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp roslib std_msgs stereo_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(flow_view src/main.cpp)

target_link_libraries(
  flow_view
  ${CMAKE_THREAD_LIBS_INIT}
  ${catkin_LIBRARIES}
)

install(TARGETS flow_view
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/   
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch                   
)   

