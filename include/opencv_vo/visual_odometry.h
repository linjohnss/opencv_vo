#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <algorithm> // for copy
#include <cstddef>
#include <ctime>
#include <ctype.h>
#include <iostream>
#include <iterator> // for ostream_iterator
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
#define MAX_NUM_FEAT 5000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2
#define DIRECT_MODE 0
#define FEATURE_MODE 1

void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1,
                        vector<Point2f> &points2, vector<uchar> &status) 
{
    vector<float> err;
    Size winSize = Size(40, 40);
    Size SPwinSize = Size(3,3);		//search window size=(2*n+1,2*n+1)
    Size zeroZone = Size(1,1);	// dead_zone size in centre=(2*n+1,2*n+1)
    TermCriteria SPcriteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    cornerSubPix(img_1, points1, SPwinSize, zeroZone, SPcriteria);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3,
                        termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        Point2f pt = points2.at(i - indexCorrection);
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

void featureDetection(Mat img, vector<Point2f> &points, int feature, Mat &descriptors) 
{
    // Mat descriptors;
    vector<KeyPoint> keypoints;
    if (feature == FEATURE_FAST) {
        // FAST algorithm
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints, points, vector<int>());
    } else if (feature == FEATURE_ORB) {
        // ORB algorithm
        Ptr<Feature2D> orb = ORB::create(MAX_NUM_FEAT);
        orb->detectAndCompute(img, Mat(), keypoints, descriptors);
        KeyPoint::convert(keypoints, points, vector<int>());
    } else if (feature == FEATURE_SHI_TOMASI) {
        // Shi-Tomasi algorithm
        goodFeaturesToTrack(img, points, MAX_NUM_FEAT, 0.01, 10);
    }
}

void findFeatureMatch(Mat img_1, Mat img_2, vector<Point2f> &points1, vector<Point2f> &points2)
{
    Mat descriptors_1;
    Mat descriptors_2;
    vector<KeyPoint> keypoints_1, keypoints_2;
    Ptr<ORB> orb = ORB::create(MAX_NUM_FEAT);
    orb->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    double minDist = 10000;

    for (int i=0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) 
            minDist = dist;
    }
    
    vector<DMatch> goodmatches;
    for (int i=0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * minDist, 30.0)) {
            goodmatches.push_back(matches[i]);
        }
    }
    
    vector<Point2f> pt_1, pt_2;
    for(int i=0; i < (int)goodmatches.size(); i++) {
        pt_1.push_back(keypoints_1[goodmatches[i].queryIdx].pt);
        pt_2.push_back(keypoints_2[goodmatches[i].trainIdx].pt);
    }

    points1 = pt_1;
    points2 = pt_2;
}

void featureMatch(vector<Point2f> &points1, vector<Point2f> &points2, Mat &descriptors_1, Mat &descriptors_2)
{
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    double minDist = 10000;

    for (int i=0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) 
            minDist = dist;
    }
    
    vector<DMatch> goodmatches;
    for (int i=0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * minDist, 30.0)) {
            goodmatches.push_back(matches[i]);
        }
    }
    
    vector<Point2f> pt_1, pt_2;
    for(int i=0; i < (int)goodmatches.size(); i++) {
        pt_1.push_back(points1[goodmatches[i].queryIdx]);
        pt_2.push_back(points2[goodmatches[i].trainIdx]);
    }

    points1 = pt_1;
    points2 = pt_2;
}