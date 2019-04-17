cmake_minimum_required(VERSION 2.8.3)
project(LMB_dispnet)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  roslib
  std_msgs
  stereo_msgs
)

find_package( PythonLibs REQUIRED )
find_package( Boost REQUIRED COMPONENTS python system )
find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
find_package( PkgConfig REQUIRED )
pkg_search_module(GLOG REQUIRED libglog)


catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp roslib std_msgs stereo_msgs
)

include_directories(
  SYSTEM ${catkin_INCLUDE_DIRS}

  SYSTEM "/usr/include/python2.7"
  "$ENV{LMB_CAFFE_DIR}/include"
  "$ENV{LMB_CAFFE_DIR}/build/src"
  SYSTEM "/usr/local/cuda/include"
  SYSTEM ${PYTHON_INCLUDE_DIRS}
  SYSTEM ${GLOG_INCLUDE_DIRS}
  SYSTEM ${Boost_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set( CMAKE_CXX_FLAGS "-W -Wall -Wextra -Wpedantic" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3" )


include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
## External dependency: CImg (http://cimg.eu)
ExternalProject_Add(
  CImg
  GIT_REPOSITORY https://github.com/dtschump/CImg.git
  SOURCE_DIR "CImg"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/CImg)


add_definitions("-Dcimg_display=0")

add_executable(dispnet src/main.cpp)

target_link_libraries (
  dispnet
  ${catkin_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${PYTHON_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  "-L/usr/local/cuda/lib64"
  "-L$ENV{LMB_CAFFE_DIR}/build/lib"
  caffe
  ${GLOG_LDFLAGS} 
)

install(TARGETS dispnet
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/                                                                                                
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch                   
)   

