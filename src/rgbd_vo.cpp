#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include "../include/opencv_vo/visual_odometry.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "opencv2/rgbd.hpp"

cv::Mat prevImage, currImage, detectImage, prevDepthImage, currDepthImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;
// std::vector<cv::DMatch> matches;
std::vector<cv::Point3f> Features3D;
cv::Mat prevDescriptors, currDescriptors;
cv::Mat E, R, t, R_f, t_f, r;
bool is_init = true;
cv::rgbd::RgbdICPOdometry *odom;
cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0);

void imageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image)
{
    try {
        detectImage = cv_bridge::toCvShare(rgb_image, sensor_msgs::image_encodings::BGR8)->image; 
        cv::cvtColor(detectImage, currImage, cv::COLOR_BGR2GRAY);
        currDepthImage = cv_bridge::toCvShare(depth_image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
        
        cv::Mat rigidTransform;
        if (!prevImage.empty() && !currImage.empty()) { 
            // findFeatureMatch(prevImage, currImage, prevFeatures, currFeatures);
            // create3DPoint(prevDepthImage, prevFeatures, Features3D, cameraMatrix);          
            std::cout << "3d-2d pairs: " << Features3D.size() << std::endl;
            solvePnP(Features3D, currFeatures, cameraMatrix, cv::Mat(), r, t, false);
            std::cout << "Yes" << std::endl;
            cv::Rodrigues(r, R);
            if (is_init == true) {
                t_f = t.clone();
                R_f = R.clone();
                is_init = false;
            }
            else {
               t_f = t_f + (R_f * t);
               R_f = R * R_f;
            }
            std::cout << "R = " << std::endl
                      << " " << R_f << std::endl
                      << std::endl;
            std::cout << "t = " << std::endl
                      << " " << t_f << std::endl
                      << std::endl;
        }

        prevImage = currImage.clone();
        prevDepthImage = currDepthImage.clone();

        cv::imshow("depth", currDepthImage); 
        cv::imshow("camera", detectImage);
        cv::waitKey(1);

    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", rgb_image->encoding.c_str());
    }
}


int main(int argc, char **argv) 
{
    ros::init(argc, argv, "rgbd_vo");
    ros::NodeHandle nh;
    cv::namedWindow("camera");
    cv::namedWindow("depth");
    cv::namedWindow("trajectory");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(rgb_sub, depth_sub, 10);
    sync.registerCallback(boost::bind(&imageCallback, _1, _2));
    // cameraMatrix = (cv::Mat_<double>(3,3) << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0);
    ros::spin();
    cv::destroyAllWindows();
}
