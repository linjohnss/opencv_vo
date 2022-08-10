#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include "../include/opencv_vo/visual_odometry.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "opencv2/rgbd.hpp"

cv::Mat prevImage, currImage, detectImage, prevDepthImage, currDepthImage;
cv::Mat E, R, t, R_f, t_f, r;
bool is_init = true;
cv::Mat traj = cv::Mat::zeros(1000, 1000, CV_8UC3);
char text[100];
cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 630.1563720703125, 0.0, 642.2313232421875,
                                             0.0, 629.5250854492188,
                                             359.1725769042969, 0.0,
                                             0.0, 1.0);
// cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0);

void imageCallback(const sensor_msgs::ImageConstPtr& rgb_image, const sensor_msgs::ImageConstPtr& depth_image)
{
    try {
        detectImage = cv_bridge::toCvShare(rgb_image, sensor_msgs::image_encodings::BGR8)->image; 
        cv::cvtColor(detectImage, currImage, cv::COLOR_BGR2GRAY);
        currDepthImage = cv_bridge::toCvShare(depth_image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
        if (!prevImage.empty() && !currImage.empty()) { 
            std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
            std::vector<cv::DMatch> matches;
            findFeatureMatch(prevImage, currImage, keypoints_1, keypoints_2, matches);
            std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;
            std::vector<cv::Point3f> pts_3d;
            std::vector<cv::Point2f> pts_2d;
            for (cv::DMatch m:matches) {
                uint32_t d = prevDepthImage.ptr<uint32_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
                if (d == 0)   // bad depth
                    continue;
                float dd = d / 50000000.0;
                cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, cameraMatrix);
                pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
                pts_2d.push_back(keypoints_2[m.trainIdx].pt);
            }
            
            solvePnP(pts_3d, pts_2d, cameraMatrix, cv::Mat(), r, t, false);
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

            
            cv::drawMatches(prevImage, keypoints_1, currImage, keypoints_2, matches, detectImage);
            int x = int(t_f.at<double>(0)) + 700;
            int y = int(t_f.at<double>(2)) + 400;
            cv::circle(traj, cv::Point(x, y), 0, cv::Scalar(255, 0, 0), 2);
            cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
            sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm",
                    t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
            cv::putText(traj, text, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN,
                        1, cv::Scalar::all(255), 1, 8);
            cv::imshow("depth", currDepthImage); 
            cv::imshow("camera", detectImage);
            cv::imshow("trajectory", traj);
            cv::waitKey(1);
        }
        prevImage = currImage.clone();
        prevDepthImage = currDepthImage.clone();
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
