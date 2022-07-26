#include <ctype.h>

#include <algorithm>  // for copy
#include <cstddef>
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#define MAX_NUM_FEAT 5000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2
#define DIRECT_MODE 0
#define FEATURE_MODE 1

void featureTracking(cv::Mat img_1, cv::Mat img_2,
                     std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2,
                     std::vector<uchar> &status) {
    std::vector<float> err;
    cv::Size winSize = cv::Size(40, 40);
    cv::Size SPwinSize = cv::Size(3, 3);  // search window size=(2*n+1,2*n+1)
    cv::Size zeroZone =
        cv::Size(1, 1);  // dead_zone size in centre=(2*n+1,2*n+1)
    cv::TermCriteria SPcriteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::cornerSubPix(img_1, points1, SPwinSize, zeroZone, SPcriteria);
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err,
                             winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        cv::Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureDetection(cv::Mat img, std::vector<cv::Point2f> &points,
                      int feature, cv::Mat &descriptors) {
    // Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    if (feature == FEATURE_FAST) {
        // FAST algorithm
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        cv::FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    } else if (feature == FEATURE_ORB) {
        // ORB algorithm
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_NUM_FEAT);
        orb->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    } else if (feature == FEATURE_SHI_TOMASI) {
        // Shi-Tomasi algorithm
        cv::goodFeaturesToTrack(img, points, MAX_NUM_FEAT, 0.01, 10);
    }
}

void findFeatureMatch(cv::Mat img_1, cv::Mat img_2,
                      std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2) {
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_NUM_FEAT);
    orb->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_HAMMING);

    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    double minDist = 10000;

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
    }

    std::vector<cv::DMatch> goodmatches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * minDist, 30.0)) {
            goodmatches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point2f> pt_1, pt_2;
    for (int i = 0; i < (int)goodmatches.size(); i++) {
        pt_1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
        pt_2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
    }

    points1 = pt_1;
    points2 = pt_2;
}

void feature2Dto3D(cv::Mat depth_1, std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point3f> &depth) {
    std::vector<cv::Point3f> d_1;
    for (int i=0; i < points1.size(); i++) {
        uint d = depth_1.ptr<unsigned int>(int(points1[i].y))[int(points1[i].x)];
        if (d == 0)
            continue;
        double dd = d / 1000.0;
        d_1.push_back(cv::Point3d(points1[i].x * dd, points1[i].y * dd, dd));
    }

    depth = d_1;
}

void featureMatch(std::vector<cv::Point2f> &points1,
                  std::vector<cv::Point2f> &points2, cv::Mat &descriptors_1,
                  cv::Mat &descriptors_2) {
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::BFMatcher::create(cv::NORM_HAMMING);

    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    double minDist = 10000;

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
    }

    std::vector<cv::DMatch> goodmatches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * minDist, 30.0)) {
            goodmatches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point2f> pt_1, pt_2;
    for (int i = 0; i < (int)goodmatches.size(); i++) {
        pt_1.push_back(points1[goodmatches[i].queryIdx]);
        pt_2.push_back(points2[goodmatches[i].trainIdx]);
    }

    points1 = pt_1;
    points2 = pt_2;
}
