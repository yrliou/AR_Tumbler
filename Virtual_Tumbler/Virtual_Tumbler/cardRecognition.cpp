#include "cardRecognition.h"

void colorFilter(cv::Mat &image){

    float RESIZE_SCALE = 0.2;
    int MAX_CARDS_DETECT = 6;
    float MIN_AREA = 1000.0;
    float MAX_AREA = 90000.0;
    
    // resize image
    cv::Mat resizeImage;
    cv::resize(image, resizeImage, cv::Size(), RESIZE_SCALE, RESIZE_SCALE);

    cv::Mat display_image;
    cv::Mat gray_image;
    cv::Mat hsv_image;
    cv::Mat canny_edge;
    cv::cvtColor(resizeImage, hsv_image, cv::COLOR_BGR2HSV);
    cv::cvtColor(resizeImage, gray_image, cv::COLOR_BGR2GRAY);
    
    //***************    keep yellow range color
    
    cv::Mat yellow_mask;
    // inRange will give 255 when inrange and 0 when not inrange
    // H: yellow is around 35, S: larger means more saturated V: larger means darker
    // OpenCV HSV range is: H: 0 to 179 S: 0 to 255 V: 0 to 255
    cv::inRange(hsv_image, cv::Scalar(10, 150, 150), cv::Scalar(50, 200, 255), yellow_mask);
    gray_image.copyTo(canny_edge, yellow_mask);
    // Result of range color is canny_edge
    
    //***************   canny edge
    cv::GaussianBlur(canny_edge, gray_image, cv::Size(5,5), 1.2, 1.2);
    cv::Canny(gray_image, canny_edge, 30, 100, 3);
    // Result of canny edge is canny_edge
    
    //**************    find contour
    cv::vector<cv::vector<cv::Point>> contours;
    cv::vector<cv::Vec4i> hierarchy;
    //findcontours will change canny_edge
    //also note that findcontours perform badly when cards on the screen edge
    cv::findContours(canny_edge, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1, cv::Point(0,0));
    
    //*************     Remove unrelated contour
    // 0. remove too small area
    // 1. sorting by area
    // 2. check whether points in the preivous contour
    // 3. limit cards card detection number by MAX_CARDS_DETECT
    cv::vector<cv::vector<cv::Point>> contours_remove_small_area;
    
    //printContourArea(contours);
    
    contours_remove_small_area = removeSmallArea(contours, MIN_AREA, MAX_AREA);
    
    std::sort(contours_remove_small_area.begin(), contours_remove_small_area.end(), compareContourAreas);
    
    //printContourArea(contours);
    cv::vector<cv::vector<cv::Point>> card_contours;
    card_contours = removeFalseContour(contours_remove_small_area, MAX_CARDS_DETECT);
    
    std::cout << "nodata" << std::endl;
    printContourArea(card_contours);
    
    plotCircle(image, RESIZE_SCALE, card_contours);
    cv::cvtColor(image, image, CV_BGR2RGBA);
    return;
    
    //printContourArea(card_contours);
    // Result of this stage is card_contours
    
    //************     Get four corners of cards
    //plotCircle(image, RESIZE_SCALE, card_contours);
    cv::vector<cv::vector<cv::Point>> card_corners;
    
    // Using HoughlineP
    // 1. transform card_contours to mat
    // 2. get lines from HoughlineP (will try probabilistic Hough transform)
    cv::vector<cv::vector<cv::Point>> card_points;
    cv::Mat corners_image;
    cv::vector<cv::Point> contour_;
    cv::vector<cv::Vec2f> lines;
    for(int i = 0; i < card_contours.size(); i++){
        
        contour_ = card_contours[i];
        corners_image = cv::Mat::zeros(resizeImage.rows, resizeImage.cols, CV_8UC1);
        
        // fill the points
        for(int j = 0; j < contour_.size();j++){
            cv::circle(corners_image, contour_[j], 0, cv::Scalar(255,255,255));
        }
        
        cv::HoughLines(corners_image, lines, 1, CV_PI/180, 10);
        
        std::cout << "HoughLinesP output is "<< lines.size() << std::endl;
        
    }
    //plotCircle(image, RESIZE_SCALE, card_contours);
    
    
    /* Approach with approx Didn't work
    cv::vector<cv::Point> approx;
    for(int i = 0; i < card_contours.size(); i++){
        cv::approxPolyDP(card_contours[i], approx, 0.01, true);
        std::cout << " card_corners index " << i << " has " << approx.size() << std::endl;
        card_corners.push_back(approx);
    }
     */
    //plotCircle(image, RESIZE_SCALE, card_corners);
    
    /*
    // draw contour
    std::cout << "How many contour " << contours.size() << std::endl;
    cv::vector<cv::Point> contour_;
    
    
    for( int i = 0; i< contours.size(); i++ )
    {
        //std::cout << "contour size is " << contours[i].size() << " index is " << i << std::endl;
        contour_ = contours[i];
        cv::Scalar color = cv::Scalar(0,0,255);
        
        for(int j = 0; j < contour_.size(); j++){
            contour_[j].x = contour_[j].x * 5;
            contour_[j].y = contour_[j].y * 5;
            cv::circle(image, contour_[j], 3, color);
        }
        //cv::drawContours( image, contours, i, color, 1, 8);
    }
    */
    //cv::cvtColor(corners_image, image, CV_GRAY2RGBA);
    cv::cvtColor(image, image, CV_BGR2RGBA);
    //cv::cvtColor(resizeImage, image, CV_BGR2RGBA);
    //cv::cvtColor(canny_edge, image, CV_GRAY2RGBA);
    //cv::cvtColor(gray_image, image, CV_GRAY2RGBA);
    
    
}

cv::vector<cv::vector<cv::Point>> removeSmallArea(cv::vector<cv::vector<cv::Point>> &contours, float MIN_AREA, float MAX_AREA){
    cv::vector<cv::vector<cv::Point>> contours_remove_small_area;
    
    
    for(int i = 0; i < contours.size(); i++){
        if(cv::contourArea(contours[i]) > MIN_AREA && cv::contourArea(contours[i]) < MAX_AREA){
            contours_remove_small_area.push_back(contours[i]);
        }
    }
    
    return contours_remove_small_area;
}

void plotCircle(cv::Mat &image, float RESIZE_SCALE, cv::vector<cv::vector<cv::Point>> contours){
    
    cv::vector<cv::Point> contour_;
    for( int i = 0; i< contours.size(); i++ ){
        //std::cout << "contour size is " << contours[i].size() << " index is " << i << std::endl;
        contour_ = contours[i];
        cv::Scalar color = cv::Scalar(0,0,255);
        
        for(int j = 0; j < contour_.size(); j++){
            contour_[j].x = contour_[j].x / RESIZE_SCALE;
            contour_[j].y = contour_[j].y / RESIZE_SCALE;
            cv::circle(image, contour_[j], 5, color, 5, 10, 0);
        }
        //cv::drawContours( image, contours, i, color, 1, 8);
    }
}

// plortCircle overloading
void plotCircle(cv::Mat &image, float RESIZE_SCALE, cv::vector<cv::Point> contour){
    
    cv::Scalar color = cv::Scalar(0,0,255);
        
    for(int j = 0; j < contour.size(); j++){
        contour[j].x = contour[j].x / RESIZE_SCALE;
        contour[j].y = contour[j].y / RESIZE_SCALE;
        cv::circle(image, contour[j], 5, color, 5, 10, 0);
    }
}

cv::vector<cv::vector<cv::Point>> removeFalseContour(cv::vector<cv::vector<cv::Point>> contours, int MAX_CARDS_DETECT){
    cv::vector<cv::vector<cv::Point>> card_contours;
    
    if (contours.size() < 2){
        return contours;
    }
    
    card_contours.push_back(contours[0]);
    for(int i = 1; i < contours.size(); i++){
        
        int success = 0;
        for(success = 0; success < card_contours.size(); success++){
            int checkwithin = cv::pointPolygonTest(card_contours[success], contours[i][0], false);
            // checkwithin, positive (inside), negative(outside), zero(on an edge)
            if(checkwithin >= 0){
                std::cout << "Really remove the contour\n\n" << std::endl;
                std::cout << "compared contour area " << cv::contourArea(contours[i]) << std::endl;
                std::cout << "remove contour area " << cv::contourArea(card_contours[success]) << std::endl;
                break;
            }
        }
        // when success is equal to contour size, it means
        // contours[i] is not within the region of card_contours
        if (success == card_contours.size()){
            std::cout << "add contour " << i << std::endl;
            card_contours.push_back(contours[i]);
        }
        
        if (card_contours.size() >= MAX_CARDS_DETECT){
            std::cout << "Too many max cards" << std::endl;
            break;
        }
    }
    
    std::cout << "return removeFalseContour" << std::endl;
    printContourArea(card_contours);
    return card_contours;
}

// sorting areas by descending order
bool compareContourAreas(cv::vector<cv::Point> contour1, cv::vector<cv::Point> contour2){
    double i = fabs(cv::contourArea(contour1));
    double j = fabs(cv::contourArea(contour2));
    
    return(i > j);
}

void printContourArea(cv::vector<cv::vector<cv::Point>> contours){
    
    std::cout << "contour size is " << contours.size() << std::endl;
    for (int i = 0; i < contours.size(); i++){
        std::cout << "index " << i << " Area is " << cv::contourArea(contours[i]) << std::endl;
    }
}
