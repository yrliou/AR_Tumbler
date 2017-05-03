#include "cardIdentify.h"
#include "armadillo"
#include <iostream>
#include <algorithm>    // std::cout


void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::ORB *orb_detector_, const char *model3dname, cv::vector<cv::vector<cv::Point>>& card_corners){
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
    /*
    K << 3043.72 <<       0 << 1196 << arma::endr
    <<       0 << 3043.72 << 1604 << arma::endr
    <<       0 <<    0    <<    1;
    */
    
        // Set camera intrinsic
        K << 1899.4 <<       0 << 978.3 << arma::endr
        <<       0 << 1897.5 << 549.7 << arma::endr
        <<       0 <<    0    <<    1;
     
    // Tau.at(0,0) += 5;
    // Tau.at(1,0) += 18;
    /*
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat pts_in_camera_frame = Omega * sphere + repmat(Tau, 1, sphere.n_cols);
    arma::fmat pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,2) / repmat(pts_2d.row(2), 3, 1);
    
    arma::fmat sphere_2d = pts_2d.rows(0,1);
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage = DrawPts(firstframe, sphere_2d, YELLOW);
    */
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
    // cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(5,24,2));
    cv::vector<cv::DMatch> matches;
    cv::vector<cv::DMatch> good_matches;
    
    matcher.match( descriptor_first, descriptor_second, matches );
    
    //-- Localize the object
    cv::vector<cv::Point2f> firstPoints;
    cv::vector<cv::Point2f> secondPoints;
    
    
    // cv::vector<cv::Point2f> goodSecondPoints;
    bool enablePolygonTest = card_corners.size() > 0;
    cv::vector<cv::Point2f> vert(4);
    if (enablePolygonTest) {
        vert[0] = card_corners[0][0];
        vert[1] = card_corners[0][1];
        vert[2] = card_corners[0][2];
        vert[3] = card_corners[0][3];
        std::cout << "vert" << vert << std::endl;
    }
    
    
    /*
    /// Get the contours
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;
    cv::Mat src_copy = second_gray.clone();
    findContours( src_copy, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    */
    
    for( int i = 0; i < matches.size(); i++ )
    {
        if (enablePolygonTest && cv::pointPolygonTest(card_corners[0], keypoints_second[ matches[i].trainIdx ].pt, false) != 1) {
            continue;
        }

        //-- Get the keypoints from matches
        firstPoints.push_back( keypoints_first[ matches[i].queryIdx ].pt );
        secondPoints.push_back( keypoints_second[ matches[i].trainIdx ].pt );
        good_matches.push_back(matches[i]);
    }
    
    std::cout << "matches numbers " << matches.size() << std::endl;
    std::cout << "good matches" << good_matches.size() << std::endl;
    DrawPts(secondframe, secondPoints, cv::Scalar(0,0,255));

    cv::Mat mask;
    // cv::Mat cardsHomography = cv::findHomography( firstPoints, secondPoints, CV_RANSAC, 6, mask);
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
    
    arma::fmat arma_homo( reinterpret_cast<float*>(cardsHomography.data), cardsHomography.rows, cardsHomography.cols );
    
    std::cout << "arma_homo\n" << arma_homo << std::endl;
    
    
    //update pts_2d

    /*
    arma::fmat pts_2d_update = arma::conv_to<arma::fmat>::from(arma_homo) * pts_2d;
    
    pts_2d_update = pts_2d_update.rows(0,2) / repmat(pts_2d_update.row(2), 3, 1);
    
    arma::fmat sphere_2d_update = pts_2d_update.rows(0,1);
    */
    /*
    arma::fmat Omega2;
    arma::fmat Tau2;
    myfit_extrinsic(arma_homo, K, Omega2, Tau2);
    
    arma::fmat pts_in_camera_frame2 = Omega2 * sphere + repmat(Tau2, 1, sphere.n_cols);
    arma::fmat pts_2d2 = K * pts_in_camera_frame2;
    pts_2d2 = pts_2d2.rows(0,2) / repmat(pts_2d2.row(2), 3, 1);
    
    arma::fmat sphere_2d_update = pts_2d2.rows(0,1);
     
    const cv::Scalar RED = cv::Scalar(0,255,255);
    cv::Mat cvImage_update = DrawPts(secondframe, sphere_2d_update, RED);
    */
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat Omega2;
    arma::fmat Tau2;
    myfit_extrinsic(arma_homo, K, Omega2, Tau2);
    arma::fmat pts_in_camera_frame = Omega2 * sphere + repmat(Tau2, 1, sphere.n_cols);
    arma::fmat pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,2) / repmat(pts_2d.row(2), 3, 1);
    
    arma::fmat sphere_2d = pts_2d.rows(0,1);
    /*
    for (int i = 0; i < sphere_2d.n_rows; i++) {
        for (int j = 0; j < sphere_2d.n_cols; j++) {
            sphere_2d.at(i, j) = sphere_2d.at(i, j) * 2;
        }
    }
    */
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage = DrawPts(secondframe, sphere_2d, YELLOW);
    // cv::drawMatches(first_gray, keypoints_first, second_gray, keypoints_second, good_matches, secondframe);
}

void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::BRISK *brisk_detector_, const char *model3dname, cv::vector<cv::vector<cv::Point>>& card_corners){
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
    /*
     K << 3043.72 <<       0 << 1196 << arma::endr
     <<       0 << 3043.72 << 1604 << arma::endr
     <<       0 <<    0    <<    1;
     */
    
    // Set camera intrinsic
    K << 1899.4 <<       0 << 978.3 << arma::endr
    <<       0 << 1897.5 << 549.7 << arma::endr
    <<       0 <<    0    <<    1;
    
    // Tau.at(0,0) += 5;
    // Tau.at(1,0) += 18;
    /*
     arma::fmat sphere; sphere.load(model3dname);
     
     arma::fmat pts_in_camera_frame = Omega * sphere + repmat(Tau, 1, sphere.n_cols);
     arma::fmat pts_2d = K * pts_in_camera_frame;
     pts_2d = pts_2d.rows(0,2) / repmat(pts_2d.row(2), 3, 1);
     
     arma::fmat sphere_2d = pts_2d.rows(0,1);
     
     const cv::Scalar YELLOW = cv::Scalar(255,255,0);
     cv::Mat cvImage = DrawPts(firstframe, sphere_2d, YELLOW);
     */
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
    
    brisk_detector_->detect(firstframe, keypoints_first);
    brisk_detector_->detect(secondframe, keypoints_second);
    brisk_detector_->compute(firstframe, keypoints_first, descriptor_first );
    brisk_detector_->compute(secondframe, keypoints_second, descriptor_second );
    
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    // cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(5,24,2));
    cv::vector<cv::DMatch> matches;
    cv::vector<cv::DMatch> good_matches;
    
    matcher.match( descriptor_first, descriptor_second, matches );
    
    //-- Localize the object
    cv::vector<cv::Point2f> firstPoints;
    cv::vector<cv::Point2f> secondPoints;
    
    
    // cv::vector<cv::Point2f> goodSecondPoints;
    bool enablePolygonTest = card_corners.size() > 0;
    cv::vector<cv::Point2f> vert(4);
    if (enablePolygonTest) {
        vert[0] = card_corners[0][0];
        vert[1] = card_corners[0][1];
        vert[2] = card_corners[0][2];
        vert[3] = card_corners[0][3];
        std::cout << "vert" << vert << std::endl;
    }
    
    
    /*
     /// Get the contours
     cv::vector<cv::vector<cv::Point> > contours;
     cv::vector<cv::Vec4i> hierarchy;
     cv::Mat src_copy = second_gray.clone();
     findContours( src_copy, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
     */
    
    for( int i = 0; i < matches.size(); i++ )
    {
        if (enablePolygonTest && cv::pointPolygonTest(card_corners[0], keypoints_second[ matches[i].trainIdx ].pt, false) != 1) {
            continue;
        }
        
        //-- Get the keypoints from matches
        firstPoints.push_back( keypoints_first[ matches[i].queryIdx ].pt );
        secondPoints.push_back( keypoints_second[ matches[i].trainIdx ].pt );
        good_matches.push_back(matches[i]);
    }
    
    std::cout << "matches numbers " << matches.size() << std::endl;
    std::cout << "good matches" << good_matches.size() << std::endl;
    DrawPts(secondframe, secondPoints, cv::Scalar(0,0,255));
    
    cv::Mat mask;
    // cv::Mat cardsHomography = cv::findHomography( firstPoints, secondPoints, CV_RANSAC, 6, mask);
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
    
    arma::fmat arma_homo( reinterpret_cast<float*>(cardsHomography.data), cardsHomography.rows, cardsHomography.cols );
    
    std::cout << "arma_homo\n" << arma_homo << std::endl;
    
    
    //update pts_2d
    
    /*
     arma::fmat pts_2d_update = arma::conv_to<arma::fmat>::from(arma_homo) * pts_2d;
     
     pts_2d_update = pts_2d_update.rows(0,2) / repmat(pts_2d_update.row(2), 3, 1);
     
     arma::fmat sphere_2d_update = pts_2d_update.rows(0,1);
     */
    /*
     arma::fmat Omega2;
     arma::fmat Tau2;
     myfit_extrinsic(arma_homo, K, Omega2, Tau2);
     
     arma::fmat pts_in_camera_frame2 = Omega2 * sphere + repmat(Tau2, 1, sphere.n_cols);
     arma::fmat pts_2d2 = K * pts_in_camera_frame2;
     pts_2d2 = pts_2d2.rows(0,2) / repmat(pts_2d2.row(2), 3, 1);
     
     arma::fmat sphere_2d_update = pts_2d2.rows(0,1);
     
     const cv::Scalar RED = cv::Scalar(0,255,255);
     cv::Mat cvImage_update = DrawPts(secondframe, sphere_2d_update, RED);
     */
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat Omega2;
    arma::fmat Tau2;
    myfit_extrinsic(arma_homo, K, Omega2, Tau2);
    arma::fmat pts_in_camera_frame = Omega2 * sphere + repmat(Tau2, 1, sphere.n_cols);
    arma::fmat pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,2) / repmat(pts_2d.row(2), 3, 1);
    
    arma::fmat sphere_2d = pts_2d.rows(0,1);
    for (int i = 0; i < sphere_2d.n_rows; i++) {
        for (int j = 0; j < sphere_2d.n_cols; j++) {
            // sphere_2d.at(i, j) = sphere_2d.at(i, j) / 2;
        }
    }
    std::cout << "sphere_2d" << sphere_2d << std::endl;
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage = DrawPts(secondframe, sphere_2d, YELLOW);
    // cv::drawMatches(first_gray, keypoints_first, second_gray, keypoints_second, good_matches, secondframe);
}


void myfit_extrinsic(const arma::fmat &H, const arma::fmat &K, arma::fmat &R, arma::fmat &t){
    
    //    R
    //    0.9994   0.0340   0.0084
    //    -0.0293   0.6782   0.7343
    //    0.0193  -0.7341   0.6788
    //
    //    t
    //    -10.7551
    //    -11.9769
    //    45.9122
    
    std::cout<<"H = "<<H<<std::endl;
    R = arma::eye<arma::fmat>(3,3);
    t = arma::ones<arma::fmat>(3,1);
    
    arma::mat H_ = arma::conv_to<arma::mat>::from(inv(K) * H);
    
    arma::mat U;
    arma::vec s;
    arma::mat V;
    arma::svd(U,s,V,H_.cols(0,1),"std");
    //    cout << "H_" << H << endl;
    
    // Estimate rotation
    arma::mat S = arma::zeros<arma::mat>(3,2);
    S(0,0) = 1; S(1,1) = 1;
    arma::mat R_ = U * S * trans(V);
    arma::vec R_lastcol = cross(R_.col(0), R_.col(1));
    for (int j = 0; j < R.n_cols; j++){
        for (int i = 0; i < R.n_rows; i++){
            if (j < 2){
                R(i,j) = (float)R_(i,j);
            } else {
                R(i,j) = (float)R_lastcol(i);
            }
        }
    }
    if (abs(det(R) - 1) > 1e2 ){
        R.col(2) = -1 * R.col(2);
    }
    //cout << "R" << R << endl;
    
    // Estimate translation
    arma::mat scale = mean(H_.cols(0,1)/R_);
    t = arma::conv_to<arma::fmat>::from(H_.col(2)/scale(0,0));
    //cout << "t" << t << endl;
    
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

cv::Mat DrawPts(cv::Mat &display_im, cv::vector<cv::Point2f> &cv_pts, const cv::Scalar &pts_clr)
{
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

cv::vector<std::string>  findcardname(cv::vector<cv::vector<cv::KeyPoint>> keypoints_database, cv::vector<cv::Mat> descriptors_database, cv::Mat grayImage, float TRACK_RESCALE, cv::ORB *orb_detector_){
    
    cv::vector<std::string> cardname;
    
    return cardname;
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
