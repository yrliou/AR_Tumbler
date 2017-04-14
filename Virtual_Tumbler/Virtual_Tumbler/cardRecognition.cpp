#include "cardRecognition.h"

void colorFilter(cv::Mat &image){


    cv::Mat display_image;
    cv::Mat gray_image;
    cv::Mat hsv_image;
    cv::Mat canny_edge;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    //keep yellow range color
    
    cv::Mat yellow_mask;
    cv::Mat mask;
    
    
    // inRange will give 255 when inrange and 0 when not inrange
    // H: yellow is around 35, S: larger means more saturated V: larger means darker
    // OpenCV HSV range is: H: 0 to 179 S: 0 to 255 V: 0 to 255
    cv::inRange(hsv_image, cv::Scalar(10, 150, 150), cv::Scalar(50, 200, 255), yellow_mask);
    gray_image.copyTo(display_image, yellow_mask);
    
    // canny edge
    //cv::blur(canny_edge, gray_image, cv::Size());
    
    
    //cv::cvtColor(display_image, image, CV_BGR2RGBA);
    cv::cvtColor(display_image, image, CV_GRAY2RGBA);
    
}
