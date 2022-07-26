#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include "../include/opencv_vo/visual_odometry.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/rgbd.hpp>

cv::Mat prevImage, currImage, detectImage, prevDepthImage, currDepthImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;
std::vector<cv::Point3f> Features3D;
cv::Mat prevDescriptors, currDescriptors;
cv::Mat E, R, t, R_f, t_f, r;
bool is_init = true;
cv::rgbd::RgbdICPOdometry *odom;
cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 718.856, 0.0, 607.1928, 0.0, 718.856,
                        185.2157, 0.0, 0.0, 1.0);

void imageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image)
{
    try {
        detectImage = cv_bridge::toCvShare(rgb_image, sensor_msgs::image_encodings::BGR8)->image; 
        cv::cvtColor(detectImage, currImage, cv::COLOR_BGR2GRAY);
        currDepthImage = cv_bridge::toCvShare(depth_image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
        
        cv::Mat rigidTransform;
        if (!prevImage.empty() && !currImage.empty()) {
            bool isSuccess = odom->compute(prevImage, prevDepthImage, cv::Mat(), currImage, currDepthImage, cv::Mat(), rigidTransform);
            R = rigidTransform(cv::Rect(0, 0, 3, 3)).clone();
            t = rigidTransform(cv::Rect(3, 0, 1, 3)).clone();
            std::cout << "Translation " <<std::endl<< t << std::endl;
            std::cout << "Rotation " <<std::endl<< R << std::endl;

            if (isSuccess == true) {
                if (is_init == true) {
                    R_f = R.clone();
                    t_f = t.clone();
                    is_init = false;
                }
                else {
                    // Update Rt
                    t_f = t_f + (R_f * t);
                    R_f = R * R_f;
                }
                ROS_INFO("Visual Odometry");
                std::cout << "Translation " <<std::endl<< t_f << std::endl;
                std::cout << "Rotation " <<std::endl<< R_f << std::endl;
            }
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
    ROS_INFO("Initializing RGBD Odometry");
    std::vector<int> iterCounts(4);
    iterCounts[0] = 7;
    iterCounts[1] = 7;
    iterCounts[2] = 7;
    iterCounts[3] = 10;

    std::vector<float> minGradMagnitudes(4);
    minGradMagnitudes[0] = 12;
    minGradMagnitudes[1] = 5;
    minGradMagnitudes[2] = 3;
    minGradMagnitudes[3] = 1;
    odom = new cv::rgbd::RgbdICPOdometry(cameraMatrix, 0.8, 4.0, 0.08, 0.09, iterCounts, minGradMagnitudes, cv::rgbd::Odometry::RIGID_BODY_MOTION);
    ros::spin();
    cv::destroyAllWindows();
}
