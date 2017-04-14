#include "cardRecognition.h"

void colorFilter(cv::Mat &image){


    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    
    //keep yellow range color
    //OpenCV HSV range is: H: 0 to 179 S: 0 to 255 V: 0 to 255
    
    cv::Mat yellow_image;
    cv::inRange(hsv_image, cv::Scalar(50, 125, 125), cv::Scalar(70, 125, 125), yellow_image);

    cv::cvtColor(hsv_image, image, cv::COLOR_HSV2BGR);
    
}
