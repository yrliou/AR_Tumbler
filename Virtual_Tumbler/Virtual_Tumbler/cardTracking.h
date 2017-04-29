#ifndef __Virtual_Tumbler_cardTracking__
#define __Virtual_Tumbler_cardTracking__

//#include <stdio.h>
#include <iostream>
//#include "armadillo" // Includes the armadillo library
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/opencv.hpp>

void cardAllFindhomography(cv::Mat &prevImage, cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, float TRACK_RESCALE, cv::Ptr<cv::DescriptorExtractor> Extractor_, cv::Ptr<cv::FeatureDetector> Detector_);

void cardAllFindhomography(cv::Mat &prevImage, cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, float TRACK_RESCALE, cv::BRISK *brisk_detector_);

void trackingCorner(cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &image, float TRACK_RESCALE);

cv::vector<cv::Point> getQuadpointOneCard(cv::vector<cv::vector<cv::Point>> &one_card_contours);

void cardTrackingOpticalFlow(cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &image, cv::Mat &prevImage, cv::vector<cv::Point2f> &prefeaturesCorners);
void featureTrack(cv::Mat &grayImage, cv::vector<cv::Point2f> &featuresCorners);

#endif /* defined(__Virtual_Tumbler_cardRTracking__) */
