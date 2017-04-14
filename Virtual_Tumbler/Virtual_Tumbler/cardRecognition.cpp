#include "cardRecognition.h"

void colorFilter(cv::Mat &image){

    // resize image
    cv::Mat resizeImage;
    cv::resize(image, resizeImage, cv::Size(), 0.2, 0.2);

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
    // Result of this stage is canny_edge
    
    //************** canny edge
    cv::GaussianBlur(canny_edge, gray_image, cv::Size(5,5), 1.2, 1.2);
    cv::Canny(gray_image, canny_edge, 30, 100, 3);
    
    //************** find contour
    
    
    //cv::cvtColor(resizeImage, image, CV_BGR2RGBA);
    cv::cvtColor(canny_edge, image, CV_GRAY2RGBA);
    
}
