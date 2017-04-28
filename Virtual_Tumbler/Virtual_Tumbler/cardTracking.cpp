#include "cardTracking.h"


void cardAllFindhomography(cv::Mat &prevImage, cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, float TRACK_RESCALE, cv::BRISK *brisk_detector_){
    
    float MIN_DIST = 20;
    std::vector<cv::KeyPoint> keypoints_prev, keypoints_current;
    
    brisk_detector_->detect(prevImage, keypoints_prev);
    brisk_detector_->detect(grayImage, keypoints_current);
    
    cv::Mat descriptors_prev, descriptors_current;
    
    brisk_detector_->compute( prevImage, keypoints_prev, descriptors_prev );
    brisk_detector_->compute( grayImage, keypoints_current, descriptors_current );
    
    /*
    if(descriptors_prev.type()!=CV_32F){
        descriptors_prev.convertTo(descriptors_prev, CV_32F);
    }
    if(descriptors_current.type()!=CV_32F){
        descriptors_current.convertTo(descriptors_current, CV_32F);
    }
     */
    
    
    // matching descriptors
    // for flann, BRIEF\ORB\FREAK have to use either LSH or Hierarchical clustering indexa
    //cv::FlannBasedMatcher matcher;
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    cv::vector<cv::DMatch> matches;
    
    matcher.match( descriptors_prev, descriptors_current, matches );
    
    //double max_dist = 0; double min_dist = 100;
    
    ////-- Quick calculation of max and min distances between keypoints
    //for( int i = 0; i < descriptors_prev.rows; i++ ){
    //    double dist = matches[i].distance;
    //    if( dist < min_dist ) min_dist = dist;
    //    if( dist > max_dist ) max_dist = dist;
    //}
    
    //std::cout << "match keyfeatures number " << matches.size() << std::endl;
    
    //std::cout << "max dist is " << max_dist << std::endl;
    //std::cout << "min dist is " << min_dist << std::endl;
    
    //cv::vector< cv::DMatch > good_matches;
    //for(int i=0;i < descriptors_prev.rows; i++){
    //    if(matches[i].distance < MIN_DIST * min_dist){
    //        good_matches.push_back(matches[i]);
    //    }
    //}
    
    //-- Localize the object
    cv::vector<cv::Point2f> previousPoints;
    cv::vector<cv::Point2f> currentPoints;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        previousPoints.push_back( keypoints_prev[ matches[i].queryIdx ].pt );
        currentPoints.push_back( keypoints_current[ matches[i].trainIdx ].pt );
    }
    std::cout << "matches numbers " << matches.size() << std::endl;
    
    
    //************************ update corners
    cv::Mat cardsHomography = cv::findHomography( previousPoints, currentPoints, CV_RANSAC );
    
    std::cout << "cards Homography\n" << cardsHomography << std::endl;
    
    
    for(int i = 0; i < card_corners.size(); i++){
        cv::vector<cv::Point2f> corners_(card_corners[i].size());
        cv::vector<cv::Point2f> corners_transform(card_corners[i].size());
        
        // IMPORTATN the card_corners is in the original scale
        // However, the image is down sampling, so the cardsHomography is downsampling as well
        // so downscaling the cardscorners (it can be omitted when we know how to multiply scaling matrix with cardsHomography)
        for(int k = 0; k < card_corners[i].size(); k++){
            corners_[k] = cv::Point2f((card_corners[i][k].x) * TRACK_RESCALE, (card_corners[i][k].y)* TRACK_RESCALE) ;
        }
        
        
        cv::perspectiveTransform(corners_, corners_transform, cardsHomography);
        
        
        std::cout << "corners_\n" << corners_ << std::endl;
        std::cout << "corners transform\n" << corners_transform << std::endl;
        //***********  Tracking card corner detection
        //trackingCorner(grayImage, corners_transform);
        
        
        
        // rescale back
        /*
        for(int j = 0; j < card_corners[i].size(); j++){
            card_corners[i][j].x = corners_transform[j].x / TRACK_RESCALE;
            card_corners[i][j].y = corners_transform[j].y / TRACK_RESCALE;
        }
        */
        for(int j = 0; j < card_corners[i].size(); j++){
            card_corners[i][j].x = corners_transform[j].x;
            card_corners[i][j].y = corners_transform[j].y;
        }
        
        
        std::cout << "final update card_corners\n" << card_corners[i] << std::endl;
    }
    
    
}

void trackingCorner(cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &image, float TRACK_RESCALE){
    
    for(int i = 0; i < card_corners.size(); i++){
        
        //cv::vector<cv::Point> card_c = card_corners[i] ;
        
        cv::Mat segment_card = cv::Mat::zeros(grayImage.rows, grayImage.cols, CV_8UC1);
        cv::Mat canny_edge;
        cv::fillConvexPoly(segment_card, card_corners[i], cv::Scalar(255,255,255));
    
        grayImage.copyTo(canny_edge, segment_card);
    
        // tracking corner detection
        //***************   canny edge
        cv::Mat gray_image;
        cv::GaussianBlur(canny_edge, gray_image, cv::Size(5,5), 1.2, 1.2);
        cv::Canny(gray_image, canny_edge, 30, 100, 3);
    
        //debug
        cv::resize(canny_edge, image, cv::Size(), 1/TRACK_RESCALE, 1/TRACK_RESCALE);
        
        //*************** adding robust morphologyEx
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,4), cv::Point(1,1));
        // After test, 4 has robust performance, but the cards can not too close to each other:
        cv::morphologyEx(canny_edge, canny_edge, 4, element);
    
        //**************    find contour
        cv::vector<cv::vector<cv::Point>> contours;
        cv::vector<cv::Vec4i> hierarchy;
    
        cv::findContours(canny_edge, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1, cv::Point(0,0));
    
        std::cout << "tracking contours " << contours.size() << std::endl;
    
        
        // card_corner scale back
        for(int j = 0; j < card_corners[i].size(); j++){
            card_corners[i][j].x = card_corners[i][j].x / TRACK_RESCALE;
            card_corners[i][j].x = card_corners[i][j].x / TRACK_RESCALE;
        }
    }
}


void cardAllFindhomography(cv::Mat &prevImage, cv::Mat &grayImage, cv::vector<cv::vector<cv::Point>> &card_corners, float TRACK_RESCALE, cv::Ptr<cv::DescriptorExtractor> Extractor_, cv::Ptr<cv::FeatureDetector> Detector_){
    
    float MIN_DIST = 5.0;
    std::vector<cv::KeyPoint> keypoints_prev, keypoints_current;
    
    Detector_->detect(prevImage, keypoints_prev);
    Detector_->detect(grayImage, keypoints_current);
    
    cv::Mat descriptors_prev, descriptors_current;
    
    Extractor_->compute( prevImage, keypoints_prev, descriptors_prev );
    Extractor_->compute( grayImage, keypoints_current, descriptors_current );
    
    // matching descriptors
    cv::FlannBasedMatcher matcher;
    cv::vector<cv::DMatch> matches;
    
    matcher.match( descriptors_prev, descriptors_current, matches );
    
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_prev.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    std::cout << "match keyfeatures number " << matches.size() << std::endl;
    
    std::cout << "max dist is " << max_dist << std::endl;
    std::cout << "min dist is " << min_dist << std::endl;
    
    cv::vector< cv::DMatch > good_matches;
    for(int i=0;i < descriptors_prev.rows; i++){
        if(matches[i].distance < MIN_DIST * min_dist){
            good_matches.push_back(matches[i]);
        }
    }
    
    //-- Localize the object
    cv::vector<cv::Point2f> previousPoints;
    cv::vector<cv::Point2f> currentPoints;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        previousPoints.push_back( keypoints_prev[ good_matches[i].queryIdx ].pt );
        currentPoints.push_back( keypoints_current[ good_matches[i].trainIdx ].pt );
    }
    std::cout << "good matches " << good_matches.size() << std::endl;
    
    
    //************************ update corners
    cv::Mat cardsHomography = cv::findHomography( previousPoints, currentPoints, CV_RANSAC );
    
    std::cout << "cards Homography\n" << cardsHomography << std::endl;
    
    
    for(int i = 0; i < card_corners.size(); i++){
        cv::vector<cv::Point2f> corners_(card_corners[i].size());
        cv::vector<cv::Point2f> corners_transform(card_corners[i].size());
        
        // IMPORTATN the card_corners is in the original scale
        // However, the image is down sampling, so the cardsHomography is downsampling as well
        // so downscaling the cardscorners (it can be omitted when we know how to multiply scaling matrix with cardsHomography)
        for(int k = 0; k < card_corners[i].size(); k++){
            corners_[k] = cv::Point2f(card_corners[i][k].x, card_corners[i][k].y) * TRACK_RESCALE;
        }
        
        cv::perspectiveTransform(corners_, corners_transform, cardsHomography);
        
        // rescale back
        for(int j = 0; j < card_corners[i].size(); j++){
            card_corners[i][j].x = corners_transform[j].x / TRACK_RESCALE;
            card_corners[i][j].y = corners_transform[j].y / TRACK_RESCALE;
        }
    }
    
}

void cardTrackingOpticalFlow(cv::vector<cv::vector<cv::Point>> &card_corners, cv::Mat &grayImage, cv::Mat &prevImage, cv::vector<cv::Point2f> &prefeaturesCorners){
    
    cv::vector<cv::Point2f> currfeaturesCorners;
    //featureTrack(image, currfeaturesCorners);
    
    //cv::Mat currgray;
    //cv::cvtColor(image, currgray, cv::COLOR_BGR2GRAY);
    //cv::Mat pregray;
    //cv::cvtColor(prevImage, pregray, cv::COLOR_BGR2GRAY);
    
    cv::vector<uchar> status;
    std::vector<float> err;
    cv::Size winSize(31,31);
    int maxLevel = 3;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    int flags = 0;
    double minEigThreshold = 0.001;
    
    //************* Using Optical Flow to calculate homography or affine
    cv::calcOpticalFlowPyrLK(prevImage, grayImage, prefeaturesCorners, currfeaturesCorners, status, err, winSize, maxLevel, termcrit, flags, minEigThreshold);
    
    std::cout << "prefeaturesCorners size " << prefeaturesCorners.size() << std::endl;
    std::cout << "currfeaturesCorners size " << currfeaturesCorners.size() << std::endl;
    
    
    //************ Transform card's corner
    //cv::findHomography();
    
    //************ Refine the corner
    
    prevImage = grayImage.clone();
    
    // how to deep copy a vector of a vector NOT Finish!!!!!!!
    prefeaturesCorners = currfeaturesCorners;
    
}

void featureTrack(cv::Mat &grayImage, cv::vector<cv::Point2f> &featuresCorners){
    
    //cv::Mat resizeImage;
    //cv::Mat grayImage;
    
    //cv::resize(image, resizeImage, cv::Size(), RESCALE, RESCALE);
    //cv::cvtColor(resizeImage, grayImage, cv::COLOR_BGR2GRAY);
    
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
