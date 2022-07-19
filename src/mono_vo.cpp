#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>

#include "../include/opencv_vo/vo_features.h"

int Mode = 1;
bool is_init = false;
cv::Mat prevImage, currImage, detectImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;
cv::Mat prevDescriptors, currDescriptors;

cv::Mat E, R, t, R_f, t_f, mask;
double scale = 1.00;
Mat traj = Mat::zeros(1000, 1000, CV_8UC3);
char text[100];
std::vector<KeyPoint> keypoints;
// Mat cameraMatrix = (Mat1d(3, 3) << 630.1563720703125, 0.0, 642.2313232421875,
//                                              0.0, 629.5250854492188, 359.1725769042969, 
//                                              0.0, 0.0, 1.0);
Mat cameraMatrix = (Mat1d(3, 3) << 718.856, 0.0, 607.1928,
                                             0.0, 718.856, 185.2157, 
                                             0.0, 0.0, 1.0);

void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
     try {
          detectImage = cv_bridge::toCvShare(msg, "bgr8")->image;
          cv::cvtColor(detectImage, currImage, COLOR_BGR2GRAY);
          if (!is_init) {
               if (!prevImage.empty() && !currImage.empty()) {
                    std::cout << "Initializing..." << std::endl;
                    if(Mode == DIRECT_MODE) {
                         std::vector<uchar> status;
                         featureDetection(prevImage, prevFeatures, FEATURE_FAST, prevDescriptors);
                         featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
                    }
                    else
                         findFeatureMatch(prevImage, currImage, prevFeatures, currFeatures);
                    E = findEssentialMat(currFeatures, prevFeatures, cameraMatrix, RANSAC,
                                        0.999, 1.0, mask);
                    recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);

                    prevImage = currImage.clone();
                    prevFeatures = currFeatures;
                    prevDescriptors = currDescriptors.clone();
                    std::cout << "R = " << std::endl << " " << R << std::endl << std::endl;
                    std::cout << "t = " << std::endl << " " << t << std::endl << std::endl;
                    R_f = R.clone();
                    t_f = t.clone();
                    is_init = true;

               } else
                    prevImage = currImage.clone();
          } else {
               std::vector<uchar> status;
               if (Mode == 0) {
                    if (prevFeatures.size() < MIN_NUM_FEAT) {
                         cout << "Number of tracked features reduced to " << prevFeatures.size()
                              << endl;
                         cout << "trigerring redection" << endl;
                         featureDetection(prevImage, prevFeatures, FEATURE_ORB, prevDescriptors);
                    }
                    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
               }
               else
                    findFeatureMatch(prevImage, currImage, prevFeatures, currFeatures);
               E = findEssentialMat(currFeatures, prevFeatures, cameraMatrix, RANSAC,
                                   0.999, 1.0, mask);
               recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);

               if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) &&
                    (t.at<double>(2) > t.at<double>(1))) {
                    t_f = t_f + scale * (R_f * t);
                    R_f = R * R_f;
               }

               prevImage = currImage.clone();
               prevFeatures = currFeatures;
               prevDescriptors = currDescriptors.clone();

               for (unsigned int i = 0; i < currFeatures.size(); i++)
                    circle(detectImage, currFeatures[i], 3, Scalar(0, 255, 0), 1, 8, 0);

               int x = int(t_f.at<double>(0)) + 700;
               int y = int(t_f.at<double>(2)) + 400;
               circle(traj, Point(x, y), 0, CV_RGB(255, 0, 0), 2);
               rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0),
                         CV_FILLED);
               sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm",
                    t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
               putText(traj, text, Point(10, 50), FONT_HERSHEY_PLAIN, 1,
                    Scalar::all(255), 1, 8);
               imshow("camera", detectImage);
               imshow("trajectory", traj);
          }
          cv::waitKey(1);
     } catch (cv_bridge::Exception &e) {
          ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
     }
}

int main(int argc, char **argv) {
     ros::init(argc, argv, "image_listener");
     ros::NodeHandle nh;
     cv::namedWindow("camera");
     cv::namedWindow("trajectory");
     cv::startWindowThread();
     image_transport::ImageTransport it(nh);
     image_transport::Subscriber sub =
          it.subscribe("/camera/color/image_raw", 1, imageCallback);
     ros::spin();
     cv::destroyWindow("view");
     cv::destroyWindow("trajectory");
}