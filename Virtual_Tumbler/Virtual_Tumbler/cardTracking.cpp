#include "cardTracking.h"

void cardTracking(cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &image, cv::Mat &prevImage, cv::vector<cv::Point2f> &prefeaturesCorners){
    
    cv::vector<cv::Point2f> currfeaturesCorners;
    //featureTrack(image, currfeaturesCorners);
    
    cv::Mat currgray;
    cv::cvtColor(image, currgray, cv::COLOR_BGR2GRAY);
    cv::Mat pregray;
    cv::cvtColor(prevImage, pregray, cv::COLOR_BGR2GRAY);
    
    cv::vector<uchar> status;
    std::vector<float> err;
    cv::Size winSize(31,31);
    int maxLevel = 3;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    int flags = 0;
    double minEigThreshold = 0.001;
    
    //************* Using Optical Flow to calculate homography or affine
    cv::calcOpticalFlowPyrLK(prevImage, image, prefeaturesCorners, currfeaturesCorners, status, err, winSize, maxLevel, termcrit, flags, minEigThreshold);
    
    std::cout << "prefeaturesCorners size " << prefeaturesCorners.size() << std::endl;
    std::cout << "currfeaturesCorners size " << currfeaturesCorners.size() << std::endl;
    
    
    //************ Transform card's corner
    //cv::findHomography();
    
    //************ Refine the corner
    
    prevImage = image.clone();
    prefeaturesCorners = currfeaturesCorners;
    
}

void featureTrack(cv::Mat &image, cv::vector<cv::Point2f> &featuresCorners, float RESCALE){
    
    cv::Mat resizeImage;
    cv::Mat grayImage;
    
    cv::resize(image, resizeImage, cv::Size(), RESCALE, RESCALE);
    cv::cvtColor(resizeImage, grayImage, cv::COLOR_BGR2GRAY);
    
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 100;
    
    cv::Size subPixWinSize(10,10);
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    
    // find good features
    cv::goodFeaturesToTrack(grayImage, featuresCorners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
    
    // corner refine
    cv::cornerSubPix(grayImage, featuresCorners, subPixWinSize, cv::Size(-1,-1), termcrit);
    
    
}
