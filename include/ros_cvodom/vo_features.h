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
#include <opencv2/flann/miniflann.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
#define MAX_NUM_FEAT 20000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2

void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1,
                     vector<Point2f> &points2, vector<uchar> &status) {
  // this function automatically gets rid of points for which tracking fails
  vector<float> err;
  Size winSize = Size(21, 21);
  TermCriteria termcrit =
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3,
                       termcrit, 0, 0.001);

  // getting rid of points for which the KLT tracking failed or those who have
  // gone outside the frame
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

void featureDetection(Mat img, vector<Point2f> &points,
                      int feature) { // uses FAST as of now, modify
                                     // parameters as necessary
  Mat descriptors;
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