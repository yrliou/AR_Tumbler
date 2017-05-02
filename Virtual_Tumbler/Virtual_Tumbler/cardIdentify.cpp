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
    
    Tau.at(0,0) += 3;
    Tau.at(1,0) -= 2;
    
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat pts_in_camera_frame = Omega * sphere + repmat(Tau, 1, sphere.n_cols);
    arma::fmat pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,2) / repmat(pts_2d.row(2), 3, 1);
    
    arma::fmat sphere_2d = pts_2d.rows(0,1);
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage = DrawPts(firstframe, sphere_2d, YELLOW);
    
    //find homography
    //gray image
    cv::Mat first_gray;
    cv::Mat second_gray;
    cv::cvtColor(firstframe, first_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(secondframe, second_gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::KeyPoint> keypoints_first;
    cv::Mat descriptor_first;
    std::vector<cv::KeyPoint> keypoints_second;
    cv::Mat descriptor_second;
    
    orb_detector_->detect(firstframe, keypoints_first);
    orb_detector_->detect(secondframe, keypoints_second);
    orb_detector_->compute(firstframe, keypoints_first, descriptor_first );
    orb_detector_->compute(secondframe, keypoints_second, descriptor_second );
    
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    cv::vector<cv::DMatch> matches;
    
    matcher.match( descriptor_first, descriptor_second, matches );
    
    //-- Localize the object
    cv::vector<cv::Point2f> firstPoints;
    cv::vector<cv::Point2f> secondPoints;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from matches
        firstPoints.push_back( keypoints_first[ matches[i].queryIdx ].pt );
        secondPoints.push_back( keypoints_second[ matches[i].trainIdx ].pt );
    }
    std::cout << "matches numbers " << matches.size() << std::endl;
    
    cv::Mat mask;
    cv::Mat cardsHomography = cv::findHomography( firstPoints, secondPoints, CV_RANSAC, 3, mask);
    
    int count_inliner = 0;
    for(int i = 0; i < mask.rows; i++){
        if (mask.at<uchar>(i)== 1){
            count_inliner += 1;
        }
    }
    
    std::cout << "projection inliner numbers " << count_inliner << std::endl;
    std::cout << "homography\n" << cardsHomography << std::endl;
    
    // cv::mat(row major) to arma::mat (column major),
    cv::transpose(cardsHomography, cardsHomography);
    std::cout << "transpose homography\n" << cardsHomography << std::endl;
    
    arma::mat arma_homo( reinterpret_cast<double*>(cardsHomography.data), cardsHomography.rows, cardsHomography.cols );
    
    std::cout << "arma_homo\n" << arma_homo << std::endl;
    
    //update pts_2d
    arma::fmat pts_2d_update = arma::conv_to<arma::fmat>::from(arma_homo) * pts_2d;
    
    pts_2d_update = pts_2d_update.rows(0,2) / repmat(pts_2d_update.row(2), 3, 1);
    
    arma::fmat sphere_2d_update = pts_2d_update.rows(0,1);
    
    const cv::Scalar RED = cv::Scalar(0,255,255);
    cv::Mat cvImage_update = DrawPts(secondframe, sphere_2d_update, RED);
    
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

cv::vector<int>  findcardname(cv::vector<cv::vector<cv::KeyPoint>> keypoints_database, cv::vector<cv::Mat> descriptors_database, cv::Mat grayImage, float TRACK_RESCALE, cv::ORB *orb_detector_, cv::vector<cv::vector<cv::Point>> card_corners, cv::vector<cv::Mat> &card_homography){
    
    // store homography
    cv::vector<cv::Mat> cardhomography(card_corners.size());
    
    cv::vector<int> cardnameIndex(card_corners.size());
    //initialize cardnameIndex
    for(int i = 0; i < cardnameIndex.size(); i++){
        cardnameIndex.push_back(-1);
    }
    
    cv::vector<cv::KeyPoint> keypoints_wholeimage;
    cv::vector<cv::vector<cv::KeyPoint>> keypoints_input(card_corners.size());
    cv::vector<cv::Mat> descriptors_input;
    
    // calculate input keypoints
    orb_detector_->detect(grayImage, keypoints_wholeimage);
    
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    cv::vector<cv::DMatch> matches;
    
    // store keypoints for input image
    for(int i = 0; i < keypoints_wholeimage.size(); i++){
        
        for(int j = 0; j < card_corners.size(); j++){
        
            int checkwithin = cv::pointPolygonTest(card_corners[j], keypoints_wholeimage[i].pt, false);
            
            // checkwithin, positive (inside), negative(outside), zero(on an edge)
            if(checkwithin >=0 ){
                keypoints_input[j].push_back(keypoints_wholeimage[i]);
            }
        }
    }
    
    
    for(int i = 0; i < keypoints_database.size(); i++){
    
        int bestinliner = 0;
        int tempinliner = 0;
        int bestindex = -1;
        cv::Mat besthomography;
        cv::Mat temphomography;
        
        for(int j = 0; j < cardnameIndex.size(); j++){
            if(cardnameIndex[j] == -1){
                temphomography = findinlinerhomo(keypoints_database[i], descriptors_database[i], keypoints_input[j], descriptors_input[j], tempinliner, matcher);
                
                if(tempinliner > bestinliner){
                    bestinliner = tempinliner;
                    besthomography = temphomography;
                    bestindex = j;
                }
            }
        }
        
        if(bestindex > -1){
            cardnameIndex[bestindex] = i;
            cardhomography[bestindex] = besthomography;
        }
    }
        
    card_homography = cardhomography;
    
    return cardnameIndex;
}

cv::Mat findinlinerhomo(std::vector<cv::KeyPoint> keypoints_database, cv::Mat descriptors_database, std::vector<cv::KeyPoint> keypoints_inputimage, cv::Mat descriptors_inputimage, int &inlinernumber, cv::FlannBasedMatcher &matcher){
    
    cv::vector<cv::DMatch> matches;
    matcher.match( descriptors_database, descriptors_inputimage, matches );
    
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
    
    cv::Mat mask;
    cv::Mat cardsHomography = cv::findHomography( databasePoints, inputPoints, CV_RANSAC, 3, mask);
    
    int count_inliner = 0;
    
    for(int i = 0; i < mask.rows; i++){
        if (mask.at<uchar>(i)== 1){
            count_inliner += 1;
        }
    }
    
    inlinernumber = count_inliner;
    
    return cardsHomography;
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
