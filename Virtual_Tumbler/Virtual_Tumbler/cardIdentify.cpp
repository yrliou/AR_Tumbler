#include "cardIdentify.h"
#include "armadillo"
#include <iostream>
#include <algorithm>    // std::cout


void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::ORB *orb_detector_, const char *model3dname){
    // Get parameters
    arma::fmat Omega;
    Omega << -0.9994 << -0.0344 << 0.0079 << arma::endr
    << 0.0292  << -0.6781 << 0.7344 << arma::endr
    << -0.0199 <<  0.7342 << 0.6787 << arma::endr;
    
    arma::fmat Tau;
    Tau << -10.6096 << arma::endr
    << -11.8144 << arma::endr
    << 45.2894 << arma::endr;
    
    arma::fmat K;
    K << 3043.72 <<       0 << 1196 << arma::endr
    <<       0 << 3043.72 << 1604 << arma::endr
    <<       0 <<    0    <<    1;
    
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat pts_in_camera_frame = Omega * sphere + repmat(Tau, 1, sphere.n_cols);
    arma::fmat pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,1) / repmat(pts_2d.row(2), 2, 1);
    
    arma::fmat sphere_2d = pts_2d.rows(0,1);
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage = DrawPts(firstframe, sphere_2d, YELLOW);
}

// Quick function to draw points on an UIImage
cv::Mat DrawPts(cv::Mat &display_im, arma::fmat &pts, const cv::Scalar &pts_clr)
{
    cv::vector<cv::Point2f> cv_pts = Arma2Points2f(pts); // Convert to vector of Point2fs
    for(int i=0; i<cv_pts.size(); i++) {
        cv::circle(display_im, cv_pts[i], 5, pts_clr,5); // Draw the points
    }
    return display_im; // Return the display image
}
// Quick function to convert Armadillo to OpenCV Points
cv::vector<cv::Point2f> Arma2Points2f(arma::fmat &pts)
{
    cv::vector<cv::Point2f> cv_pts;
    for(int i=0; i<pts.n_cols; i++) {
        cv_pts.push_back(cv::Point2f(pts(0,i), pts(1,i))); // Add points
    }
    return cv_pts; // Return the vector of OpenCV points
}

cv::Mat homographyinliner( std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, cv::Mat card_image , int &inlinernumber, cv::ORB *orb_detector_, cv::Mat &database_image){

//cv::Mat homographyinliner( std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, cv::Mat card_image , int &inlinernumber, cv::BRISK *brisk_detector_, cv::Mat &database_image){
    
    float RESCALE = 0.6;
    cv::Mat grayinput;
    cv::GaussianBlur(card_image, grayinput, cv::Size(5,5), 1.2, 1.2);
    cv::resize(grayinput, grayinput, cv::Size(), RESCALE, RESCALE);
    cv::cvtColor(grayinput, grayinput, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::KeyPoint> keypoints_inputimage;
    cv::Mat descriptor_inputimage;
    
    orb_detector_->detect(card_image, keypoints_inputimage);
    orb_detector_->compute(card_image, keypoints_inputimage, descriptor_inputimage );
    
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    cv::vector<cv::DMatch> matches;
    
    matcher.match( descriptors_database, descriptor_inputimage, matches );
    
    //-- Localize the object
    cv::vector<cv::Point2f> databasePoints;
    cv::vector<cv::Point2f> inputPoints;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from matches
        databasePoints.push_back( keypoints_database[ matches[i].queryIdx ].pt );
        inputPoints.push_back( keypoints_inputimage[ matches[i].trainIdx ].pt );
    }
    std::cout << "matches numbers " << matches.size() << std::endl;
    
    //cv::vector<uchar> mask(matches.size());
    cv::Mat mask;
    cv::Mat cardsHomography = cv::findHomography( databasePoints, inputPoints, CV_RANSAC, 3, mask);
    //cv::Mat cardsHomography = cv::findHomography( databasePoints, inputPoints, CV_RANSAC);
    
    //for mask, inliner is marked as 255
    /*
    for(int i = 0; i < mask.size(); i++){
        std::cout << mask[i] << std::endl;
    }
     */
    //inlinernumber = std::count (mask.begin(), mask.end(), (uchar)255);
    //std::cout << "mask\n" << mask << std::endl;
    int count_inliner = 0;
    
    for(int i = 0; i < mask.rows; i++){
        if (mask.at<uchar>(i)== 1){
            count_inliner += 1;
        }
    }
    
    inlinernumber = count_inliner;
    
    // draw features match
    cv::Mat img_matches;
    drawMatches( database_image, keypoints_database, grayinput, keypoints_inputimage,
                matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    std::cout << "cardsHomography is\n" << cardsHomography <<std::endl;
    //return img_matches;
    return img_matches;
}
