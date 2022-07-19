#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#define MAX_FRAME 3500

int main(int argc, char **argv) {
    ros::init(argc, argv, "kitti_publisher");
    ros::NodeHandle nh;
    
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/color/image_raw", 1);

    ros::Rate loop_rate(10);
    char filename[200];
    int numFrame=1000;
    while (nh.ok()) {
        if(numFrame < MAX_FRAME)
            numFrame++;
        else
            break;
        sprintf(filename, "/home/point/00/image_0/%06d.png", numFrame); 
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);    
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        pub.publish(msg);
        ROS_INFO("publish sequence : %06d.png", numFrame);
        ros::spinOnce();
        loop_rate.sleep();
    }
}