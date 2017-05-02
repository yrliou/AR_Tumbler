#ifndef __Virtual_Tumbler_cardIdentify__
#define __Virtual_Tumbler_cardIdentify__

#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/opencv.hpp>
#include "armadillo"

cv::Mat homographyinliner( std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, cv::Mat card_image , int &inlinernumber, cv::ORB *orb_detector_, cv::Mat &database_image);

//cv::Mat homographyinliner( std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, cv::Mat card_image , int &inlinernumber, cv::BRISK *brisk_detector_, cv::Mat &database_image);

cv::Mat DrawPts(cv::Mat &display_im, arma::fmat &pts, const cv::Scalar &pts_clr);
cv::vector<cv::Point2f> Arma2Points2f(arma::fmat &pts);

void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::ORB *orb_detector_, const char *model3dname);

cv::vector<int>  findcardname(cv::vector<cv::vector<cv::KeyPoint>> keypoints_database, cv::vector<cv::Mat> descriptors_database, cv::Mat grayImage, float TRACK_RESCALE, cv::ORB *orb_detector_, cv::vector<cv::vector<cv::Point>> card_corners, cv::vector<cv::Mat> &card_homography);

cv::Mat findinlinerhomo(std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, std::vector<cv::KeyPoint> keypoints_inputimage, cv::Mat descriptors_inputimage, int &inlinernumber, cv::FlannBasedMatcher &matcher);

#endif
