cmake_minimum_required(VERSION 3.0.2)
project(opencv_vo)
find_package( OpenCV 3.0 REQUIRED )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES opencv_vo
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(mono_vo src/mono_vo.cpp)

target_link_libraries(mono_vo
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(kitti_publisher src/kitti_publisher.cpp)

target_link_libraries(kitti_publisher
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)