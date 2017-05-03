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

cv::Mat DrawPts(cv::Mat &display_im, cv::vector<cv::Point2f> &cv_pts, const cv::Scalar &pts_clr);

cv::vector<cv::Point2f> Arma2Points2f(arma::fmat &pts);

void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::ORB *orb_detector_, const char *model3dname, cv::vector<cv::vector<cv::Point>>& card_corners);
void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::BRISK *brisk_detector_, const char *model3dname, cv::vector<cv::vector<cv::Point>>& card_corners);
void myfit_extrinsic(const arma::fmat &H, const arma::fmat &K, arma::fmat &R, arma::fmat &t);

cv::vector<std::string>  findcardname(cv::vector<cv::vector<cv::KeyPoint>> keypoints_database, cv::vector<cv::Mat> descriptors_database, cv::Mat grayImage, float TRACK_RESCALE, cv::ORB *orb_detector_);

#endif
