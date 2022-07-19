# OpenCV Visual Odometry
[![Demo video](https://i.imgur.com/NufVlkx.gif)](https://youtu.be/zcA-jLR3i3s)

## Overview
A ROS package for visual odometry based on OpenCV function:
- ROS node subscribe image messages for real time working
- ORB feature detection

## Installation
### Dependencies
This software is built on the Robotic Operating System ([ROS](http://wiki.ros.org/ROS/Tutorials)), which needs to be installed first. 
- [OpenCV](https://opencv.org/) >= 3.3

### Building
Clone the repository and catkin_make:
```shell
cd catkin_ws/src
git clone https://github.com/linjohnss/opencv_vo.git
cd ../
catkin_make
source devel/setup.bash
```

## Running opencv_vo with your camera
Change camera metrix for yor camera
```cpp=25
Mat cameraMatrix = (Mat1d(3, 3) << 718.856, 0.0, 607.1928,
                                   0.0, 718.856, 185.2157, 
                                   0.0, 0.0, 1.0);
```
```shell
rosrun opencv_vo mono_vo
```

## KITTI Example
You can use `kitti_publisher` to publish [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to ROS image msgs
```shell
rosrun opencv_vo kitti_publisher
```

