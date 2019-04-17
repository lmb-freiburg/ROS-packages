cmake_minimum_required(VERSION 2.8.3)
project(LMB_generic_images_feeder)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  roslib
  std_msgs
)

find_package( Threads REQUIRED )
find_package( PNG REQUIRED )

set( CMAKE_CXX_FLAGS "-W -Wall -Wextra -std=c++11" )
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
## External dependency: C++ pacemaker (https://github.com/nikolausmayer/cpp-pacemaker)
ExternalProject_Add(
  cpp-pacemaker
  GIT_REPOSITORY https://github.com/nikolausmayer/cpp-pacemaker
  SOURCE_DIR "cpp-pacemaker"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/cpp-pacemaker)


add_definitions("-Dcimg_display=0")
add_definitions("-Dcimg_use_png")

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp roslib std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(images_feeder src/main.cpp)

target_link_libraries(
  images_feeder
  ${CMAKE_THREAD_LIBS_INIT}
  ${catkin_LIBRARIES}
  ${PNG_LIBRARIES}
)

install(TARGETS images_feeder
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
