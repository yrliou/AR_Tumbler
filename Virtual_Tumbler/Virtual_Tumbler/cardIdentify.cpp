#include "cardIdentify.h"
#include "armadillo"
#include <iostream>
#include <algorithm>    // std::cout

void projectModel(cv::Mat &prevImag, cv::Mat &grayImage, cv::Mat &image, cv::vector<cv::vector<cv::Point>> card_corners, cv::ORB *orb_detector_, float PRORESCALE, const char *model3dname){
    
    arma::fmat sphere; sphere.load(model3dname);
    
    arma::fmat ipodK;
    ipodK<< 1899.4 << 0       << 978.3 << arma::endr
    << 0      << 1897.5  << 549.7 << arma::endr
    << 0      << 0       << 1     ;
    
    arma::fmat H_original;
    H_original << -7.1717e-02 << 1.8066e-02 << -5.1062e-01 << arma::endr
    << 1.3230e-03 <<-2.0742e-02 << -8.5637e-01 << arma::endr
    << -4.5728e-07 << 1.7184e-05 << -1.0572e-03;
    
    std::vector<cv::KeyPoint> keypoints_prev;
    cv::Mat descriptor_prev;
    std::vector<cv::KeyPoint> keypoints_curr;
    cv::Mat descriptor_curr;
    
    orb_detector_->detect(prevImag, keypoints_prev);
    orb_detector_->detect(grayImage, keypoints_curr);
    
    orb_detector_->compute(prevImag, keypoints_prev, descriptor_prev );
    orb_detector_->compute(grayImage, keypoints_curr, descriptor_curr );
    
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
    cv::vector<cv::DMatch> matches;
    
    matcher.match( descriptor_prev, descriptor_curr, matches );
    
    //-- Localize the object
    cv::vector<cv::Point2f> firstPoints;
    cv::vector<cv::Point2f> secondPoints;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from matches
        firstPoints.push_back( keypoints_prev[ matches[i].queryIdx ].pt );
        secondPoints.push_back( keypoints_curr[ matches[i].trainIdx ].pt );
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
    
    // cv::Mat to armadillo fmat
    cv::transpose(cardsHomography, cardsHomography);
    
    arma::mat arma_homo( reinterpret_cast<double*>(cardsHomography.data), cardsHomography.rows, cardsHomography.cols );
    arma::fmat arma_homo_fmat = arma::conv_to<arma::fmat>::from(arma_homo);
    
    arma::fmat R_homo;
    arma::fmat t_homo;
    
    arma::fmat secondhomo = arma_homo_fmat * H_original;
    
    myfit_extrinsic(secondhomo, ipodK, R_homo, t_homo);
    arma::fmat pts_2d = myproj_extrinsic(sphere, ipodK, R_homo, t_homo);
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage2 = DrawPts(image, pts_2d, YELLOW);
    
}

void projectImageTest(cv::Mat &firstframe, cv::Mat &secondframe, cv::ORB *orb_detector_, const char *model3dname, cv::Mat &magnemitedatabase, cv::Mat &magnemitetest2){
    
    
    // read reference image
    arma::fmat ipodK;
    ipodK<< 1899.4 << 0       << 978.3 << arma::endr
         << 0      << 1897.5  << 549.7 << arma::endr
         << 0      << 0       << 1     ;
    
    arma::fmat K;
    K << 3043.72 <<       0 << 1196 << arma::endr
    <<       0 << 3043.72 << 1604 << arma::endr
    <<       0 <<    0    <<    1;
    
    arma::fmat H;
    H << -7.1717e-02 << 1.8066e-02 << -5.1062e-01 << arma::endr
    << 1.3230e-03 <<-2.0742e-02 << -8.5637e-01 << arma::endr
    << -4.5728e-07 << 1.7184e-05 << -1.0572e-03;
    
    arma::fmat sphere; sphere.load(model3dname);
    arma::fmat z_offset;
    z_offset << 0 << arma::endr << 0 << arma::endr << 3.4290;
    arma::fmat xy_offset;
    xy_offset << 320 << arma::endr << 640 << arma::endr << 0;
    sphere = (sphere + repmat(z_offset, 1, 961))*20 + repmat(xy_offset, 1, 961);
    
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
    //std::cout << "homography\n" << cardsHomography << std::endl;
    std::vector<cv::DMatch> inlinerMatch;
    for (size_t i = 0; i < mask.rows; ++i) {
        uchar *inliner = mask.ptr<uchar>(i);
        if (inliner[0] == 1) {
            inlinerMatch.push_back(matches[i]);
        }
    }
    drawMatches( firstframe, keypoints_first, grayinput, keypoints_inputimage,
                matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    
    // cv::mat(row major) to arma::mat (column major),
    cv::transpose(cardsHomography, cardsHomography);
    //std::cout << "transpose homography\n" << cardsHomography << std::endl;
    
    arma::mat arma_homo( reinterpret_cast<double*>(cardsHomography.data), cardsHomography.rows, cardsHomography.cols );
    arma::fmat arma_homo_fmat = arma::conv_to<arma::fmat>::from(arma_homo);
    
    std::cout << "arma_homo\n" << arma_homo << std::endl;
    
    //update pts_2d
    arma::fmat R_homo;
    arma::fmat t_homo;
    
    //arma::fmat secondhomo = arma_homo_fmat * H;
    
    myfit_extrinsic(arma_homo_fmat, ipodK, R_homo, t_homo);
    arma::fmat pts_2d = myproj_extrinsic(sphere, ipodK, R_homo, t_homo);
    
    const cv::Scalar YELLOW = cv::Scalar(255,255,0);
    cv::Mat cvImage2 = DrawPts(secondframe, pts_2d, YELLOW);
    std::cout << "Result\n" << pts_2d << std::endl;
    
}

arma::fmat myproj_extrinsic(const arma::fmat &pts_3d, const arma::fmat &K,
                            const arma::fmat &R, const arma::fmat &t){
    arma::fmat pts_2d;
    
    if (pts_3d.n_rows != 3 || K.n_rows != 3 || K.n_cols != 3 ||
        R.n_rows != 3 || R.n_cols != 3 || t.n_rows != 3 || t.n_cols != 1){
        std::cerr << "myproj_extrinsic bad input dimensions!" << std::endl;
        return pts_2d;
    }
    
    arma::fmat pts_in_camera_frame = R * pts_3d + repmat(t, 1, pts_3d.n_cols);
    pts_2d = K * pts_in_camera_frame;
    pts_2d = pts_2d.rows(0,1) / repmat(pts_2d.row(2), 2, 1);
    
    return pts_2d;
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
    
    //std::cout<<"H = "<<H<<std::endl;
    R = arma::eye<arma::fmat>(3,3);
    t = arma::ones<arma::fmat>(3,1);
    
    arma::mat H_ = arma::conv_to<arma::mat>::from(arma::inv(K) * H);
    
    //std::cout << "H_\n" << H_ <<std::endl;
    //std::cout << "inv(K)\n" << arma::inv(K) <<std::endl;
    //std::cout << "inv(K)*H\n" << arma::inv(K) * H <<std::endl;
    
    arma::mat U;
    arma::vec s;
    arma::mat V;
    arma::svd(U,s,V,H_.cols(0,1),"std");
    //cout << "H_" << H << endl;
    
    // Estimate rotation
    arma::mat S = arma::zeros<arma::mat>(3,2);
    S(0,0) = 1; S(1,1) = 1;
    arma::mat R_ = U * S * arma::trans(V);
    arma::vec R_lastcol = arma::cross(R_.col(0), R_.col(1));
    for (int j = 0; j < R.n_cols; j++){
        for (int i = 0; i < R.n_rows; i++){
            if (j < 2){
                R(i,j) = (float)R_(i,j);
            } else {
                R(i,j) = (float)R_lastcol(i);
            }
        }
    }
    if (abs(arma::det(R) - 1) > 1e2 ){
        R.col(2) = -1 * R.col(2);
    }
    //cout << "R" << R << endl;
    
    // Estimate translation
    arma::mat scale = arma::mean(H_.cols(0,1)/R_);
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
