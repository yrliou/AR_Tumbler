#ifndef __Virtual_Tumbler_cardRecognition__
#define __Virtual_Tumbler_cardRecognition__

//#include <stdio.h>
#include <iostream>
//#include "armadillo" // Includes the armadillo library
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/opencv.hpp>

cv::vector<cv::vector<cv::Point>> cardRecognition(cv::Mat &image);
bool compareContourAreas(cv::vector<cv::Point> contour1, cv::vector<cv::Point> contour2);
void printContourArea(cv::vector<cv::vector<cv::Point>> contours);
cv::vector<cv::vector<cv::Point>> removeFalseContour(cv::vector<cv::vector<cv::Point>> contours, int MAX_CARDS_DETECT);
void plotCircle(cv::Mat &image, float RESIZE_SCALE, cv::vector<cv::vector<cv::Point>> contours);

cv::vector<cv::vector<cv::Point>> removeSmallArea(cv::vector<cv::vector<cv::Point>> &contours, float MIN_AREA, float MAX_AREA);
void plotCircle(cv::Mat &image, float RESIZE_SCALE, cv::vector<cv::Point> contour);
cv::vector<cv::vector<cv::Point>> getQuadpointHough(cv::vector<cv::vector<cv::Point>> &contours, cv::Mat &resizeImage, float RESIZE_SCALE, cv::Mat &image);

cv::vector<cv::vector<cv::Point>> getQuadpointApprox(cv::vector<cv::vector<cv::Point>> &card_contours);

void rescalePoints(cv::vector<cv::vector<cv::Point>> &card_corners, float RESIZE_SCALE);

#endif /* defined(__Virtual_Tumbler_cardRecognition__) */
