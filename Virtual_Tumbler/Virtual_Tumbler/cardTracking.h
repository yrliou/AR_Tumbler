#ifndef __Virtual_Tumbler_cardTracking__
#define __Virtual_Tumbler_cardRTracking__

//#include <stdio.h>
#include <iostream>
//#include "armadillo" // Includes the armadillo library
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/opencv.hpp>

void cardTracking(cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &image, cv::Mat &prevImage, cv::vector<cv::Point2f> &prefeaturesCorners);
void featureTrack(cv::Mat &image, cv::vector<cv::Point2f> &featuresCorners, float RESCALE);

#endif /* defined(__Virtual_Tumbler_cardRTracking__) */
