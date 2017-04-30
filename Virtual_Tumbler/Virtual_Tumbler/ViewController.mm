//
//  ViewController.m
//  Virtual_Tumbler
//
//  Created by Sam on 2017/4/7.
//  Copyright © 2017年 Sam. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#include <stdlib.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "cardRecognition.h"
#include "cardTracking.h"
#include "armadillo"
#include "myfit.h"
#endif

@interface ViewController (){
    // show FPS
    UITextView *fpsView_;
    int64 curr_time_;
    
    // show image
    UIImageView *imageView_;
    
    // card tracking and recognition
    int card_recognition; // count for going to card recognition
    cv::vector<cv::vector<cv::Point>> card_corners;
    cv::Mat prevImage; // store the previous image for calculating homography
    float TRACK_RESCALE;
    
    // Used for homography
    cv::BRISK *brisk_detector_;
    
    arma::fmat K;
}

@end

@implementation ViewController

// @synthesize videoCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Choose which mode
    //int VideoStream = 0; // Video
    //int VideoStream = 1; // card Tracking
    int VideoStream = 2; // card Recognition
    
    TRACK_RESCALE = 0.50;
    
    //setup Brisk descriptor and detector
    int thresh = 30;
    int octaves = 3;
    float patternSacle = 1.0f;
    brisk_detector_ = new cv::BRISK(thresh, octaves, patternSacle);
    
    // K << 3043.72 <<       0 << 1196 << arma::endr
    // <<       0 << 3043.72 << 1604 << arma::endr
    // <<       0 <<    0    <<    1;
    K << 1899.4 <<       0 << 978.3 << arma::endr
    <<       0 << 1897.5 << 549.7 << arma::endr
    <<       0 <<    0    <<    1;
    
    [self VideoStillImage:VideoStream];
}

- (void) VideoStillImage:(int) VideoStream{
    
    if (VideoStream == 0){
        // [self cameraSetup];
        card_recognition = 0;
        // [videoCamera start];
    }
    else if(VideoStream == 1){
        // debug tracking
        
        // read image_test_1card image as the first frame
        UIImage *first_frame = [UIImage imageNamed:@"first_frame.png"];
        if(first_frame == nil) {
            std::cout << "Cannot read in the file first_frame.png!!" << std::endl;
            return;
        }
        
        UIImage *second_frame = [UIImage imageNamed:@"second_frame.png"];
        if(second_frame == nil) {
            std::cout << "Cannot read in the file second_frame.png!!" << std::endl;
            return;
        }
        
        UIImage *third_frame = [UIImage imageNamed:@"third_frame.png"];
        if(third_frame == nil) {
            std::cout << "Cannot read in the file third_frame.png!!" << std::endl;
            return;
        }
        
        
        UIImage *inputImageFirst = first_frame;
        // set the ImageView_ for still image
        [self showImage:inputImageFirst];
        
        cv::Mat cvImageFirst = [self cvMatFromUIImage:inputImageFirst];
        
        // process image first - card recognition
        cv::cvtColor(cvImageFirst, cvImageFirst, CV_RGB2BGR);
        cv::vector<cv::vector<cv::Point>> card_cornersFirst;
        card_cornersFirst = cardRecognition(cvImageFirst);
        //std::cout << "size of card_cornersFirst " << card_cornersFirst.size() << std::endl;
        [self plotCircle:cvImageFirst points:card_cornersFirst];
        cv::cvtColor(cvImageFirst, cvImageFirst, CV_BGR2RGB);
        
        // process second image
        cv::Mat cvImageSecond = [self cvMatFromUIImage:second_frame];
        cv::cvtColor(cvImageSecond, cvImageSecond, CV_RGB2BGR);
        cv::Mat grayImageSecond;
        cv::Mat colorImageSecond;
        cv::Mat resizeImageSecond;
        cv::GaussianBlur(cvImageSecond, resizeImageSecond, cv::Size(5,5), 1.0, 1.0);
        cv::resize(resizeImageSecond, colorImageSecond, cv::Size(), TRACK_RESCALE, TRACK_RESCALE);
        cv::cvtColor(colorImageSecond, grayImageSecond, cv::COLOR_BGR2GRAY);
        
        cv::Mat grayImageFirst;
        cv::Mat resizeImageFirst;
        cv::cvtColor(cvImageFirst, grayImageFirst, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayImageFirst, resizeImageFirst, cv::Size(5,5), 1.0, 1.0);
        cv::resize(resizeImageFirst, grayImageFirst, cv::Size(), TRACK_RESCALE, TRACK_RESCALE);
        
        std::cout << "TRACK_RESCALE is " << TRACK_RESCALE << std::endl;
        
        // tracking
        cardAllFindhomography(grayImageFirst, grayImageSecond, card_cornersFirst, TRACK_RESCALE, brisk_detector_);
        trackingCorner(colorImageSecond, card_cornersFirst, cvImageSecond, TRACK_RESCALE);
        [self plotCircle:cvImageSecond points:card_cornersFirst];
        cv::cvtColor(cvImageSecond, cvImageSecond, CV_BGR2RGB);
        
        // process third image
        cv::Mat cvImageThird = [self cvMatFromUIImage:third_frame];
        cv::cvtColor(cvImageThird, cvImageThird, CV_RGB2BGR);
        cv::Mat grayImageThird;
        cv::Mat colorImageThird;
        cv::Mat resizeImageThird;
        cv::GaussianBlur(cvImageThird, resizeImageThird, cv::Size(5,5), 1.0, 1.0);
        cv::resize(resizeImageThird, colorImageThird, cv::Size(), TRACK_RESCALE, TRACK_RESCALE);
        cv::cvtColor(colorImageThird, grayImageThird, cv::COLOR_BGR2GRAY);
        
        //tracking
        cardAllFindhomography(grayImageSecond, grayImageThird, card_cornersFirst, TRACK_RESCALE, brisk_detector_);
        trackingCorner(colorImageThird, card_cornersFirst, cvImageThird, TRACK_RESCALE);
        //std::cout << "size of card_cornersFirst " << card_cornersFirst.size() << "\n" << card_cornersFirst[0] << "\n" << card_cornersFirst[1] << std::endl;
        [self plotCircle:cvImageThird points:card_cornersFirst];
        cv::cvtColor(cvImageThird, cvImageThird, CV_BGR2RGB);
        
        // show on the screen
        imageView_.image = [self UIImageFromCVMat:cvImageThird];
    }
    else{
        // debug still image
        // read image_test_1card image
        UIImage *image_test_1card = [UIImage imageNamed:@"image_test_1card.jpg"];
        if(image_test_1card == nil) std::cout << "Cannot read in the file image_test_1card.jpg!!" << std::endl;
        
        // read image_test_3card image
        UIImage *image_test_3cards = [UIImage imageNamed:@"image_test_3cards.jpg"];
        if(image_test_3cards == nil) std::cout << "Cannot read in the file image_test_3cards.jpg!!" << std::endl;
        
        // read image_test_many image
        UIImage *image_test_many = [UIImage imageNamed:@"image_test_many.jpg"];
        if(image_test_many == nil) std::cout << "Cannot read in the file image_test_many.jpg!!" << std::endl;
        
        // read image_3cards_half
        UIImage *image_3cards_half = [UIImage imageNamed:@"3cards_half.jpg"];
        if(image_3cards_half == nil) std::cout << "Cannot read in the file 3cards_half.jpg!!" << std::endl;
        
        // read image_3cards_hand
        UIImage *image_3cards_hand = [UIImage imageNamed:@"3cards_with_hand.jpg"];
        if(image_3cards_hand == nil) std::cout << "Cannot read in the file 3cards_with_hand.jpg!!" << std::endl;
        
        // read image_4cards
        UIImage *image_4cards = [UIImage imageNamed:@"image_test_4cards.jpg"];
        if(image_4cards == nil) std::cout << "Cannot read in the file image_test_4cards.jpg!!" << std::endl;
        
        // read image_test_many image
        UIImage *image_test_3cards_dark = [UIImage imageNamed:@"image_test_3cards_dark.jpg"];
        if(image_test_many == nil) std::cout << "Cannot read in the file image_test_3cards_dark.jpg!!" << std::endl;
        
        // read test_1card
        UIImage *test_1card = [UIImage imageNamed:@"test_1card.jpg"];
        if(test_1card == nil) std::cout << "Cannot read in the file test_1card.jpg!!" << std::endl;
        
        // read card from DB
        UIImage *test_1card_data = [UIImage imageNamed:@"test_1card_data.jpg"];
        if(test_1card_data == nil) std::cout << "Cannot read in the file test_1card_data.jpg!!" << std::endl;
        
        //UIImage *inputImage = image_4cards;
        //UIImage *inputImage = image_3cards_hand;
        //UIImage *inputImage = image_3cards_half;
        //UIImage *inputImage = image_test_3cards;
        //UIImage *inputImage = image_test_3cards_dark;
        // UIImage *inputImage = image_test_1card;
        UIImage *inputImage = test_1card;
        
        // set the ImageView_ for still image
        [self showImage:inputImage];
        
        cv::Mat cvCardSceneImg = [self cvMatFromUIImage:test_1card];
        cv::Mat cvCardDBImg = [self cvMatFromUIImage:test_1card_data];
        
        cvtColor(cvCardSceneImg, cvCardSceneImg, CV_RGBA2GRAY);
        cvtColor(cvCardDBImg, cvCardDBImg, CV_RGBA2GRAY);
        
        cv::Mat H_cv = cardFindhomography(cvCardDBImg, cvCardSceneImg, 1.0, brisk_detector_);
        
        cv::Mat cvImage = [self cvMatFromUIImage:inputImage];
        // cv::Mat cvImage = [self cvMatFromUIImage:test_1card_data];
        /*
        cv::Mat processImage;
        cv::cvtColor(cvImage, processImage, CV_RGBA2BGR);
        
        // process image
        cv::vector<cv::vector<cv::Point>> card_corners;
        card_corners = cardRecognition(processImage);
        
        [self plotCircle:processImage points:card_corners];
        cv::cvtColor(processImage, processImage, CV_BGR2RGBA);
        */
        // Load the 3D sphere points (dimensions of ball are in cm)
        NSString *str = [[NSBundle mainBundle] pathForResource:@"sphere" ofType:@"txt"];
        const char *SphereName = [str UTF8String]; // Convert to const char *
        arma::fmat sphere;sphere.load(SphereName); // Load the Sphere into memory should be 3xN
        
        cv::Mat I = (cv::Mat_<float>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat k = (cv::Mat_<float>(3,3) << 1899.4, 0, 978.3, 0, 1897.5, 549.7, 0, 0, 1);
        // cv::Mat k = (cv::Mat_<float>(3,3) << 3043.72, 0, 1196, 0, 3043.72, 1604, 0, 0, 1);
        
        // test using identity as first extrinsic
        // H_cv = k * I;
        
        std::cout << H_cv << std::endl;
        
        // cv::mat to arma
        cv::Mat H_cv_transpose;
        cv::transpose(H_cv, H_cv_transpose);
        arma::fmat arma_mat(reinterpret_cast<float*>(H_cv_transpose.data), H_cv_transpose.rows, H_cv_transpose.cols);
        
        // project using homography
        arma::fmat X = myproj_homography(sphere, arma_mat);
        const cv::Scalar YELLOW = cv::Scalar(255,255,0);
        cvImage = DrawPts(cvImage, X, YELLOW);
        
        // Finally setup the view to display
        imageView_.image = [self UIImageFromCVMat:cvImage];
        
        // show on the screen
        // imageView_.image = [self UIImageFromCVMat:processImage];
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

// Function to run apply image on
- (void) processImage:(cv::Mat &)image
{
    // Card Recognition
    /*
    card_corners = cardRecognition(image);
    [self plotCircle:image points:card_corners];
    */
    
    
    // blur image before downsampling
    cv::Mat colorImage;
    cv::Mat grayImage;
    cv::Mat resizeImage;
    cv::GaussianBlur(image, resizeImage, cv::Size(5,5), 1.0, 1.0);
    //cv::medianBlur(image, resizeImage, 5);
    cv::resize(resizeImage, colorImage, cv::Size(), TRACK_RESCALE, TRACK_RESCALE);
    
    if(card_recognition < 60){
        card_corners = cardRecognition(image);
        cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
        prevImage = grayImage.clone();
        
        //plot circle
        [self plotCircle:image points:card_corners];
        
        card_recognition += 1;
    }
    else{
        // not yet finish can move card independently
        //cardTracking(card_corners, grayImage, prevImage, prefeaturesCorners);
        
        // card can't move independently, only move camera
        cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
        
        cardAllFindhomography(prevImage, grayImage, card_corners, TRACK_RESCALE, brisk_detector_);
        trackingCorner(colorImage, card_corners, image, TRACK_RESCALE);
        
        prevImage = grayImage.clone();
        [self plotCircle:image points:card_corners];
    }
    
    // show FPS
    [self showFPS];
}


- (void) plotCircle:(cv::Mat &)image points:(cv::vector<cv::vector<cv::Point>>) contours{
    
    cv::Scalar color = cv::Scalar(0,0,255);
    
    cv::vector<cv::Point> contour_;
    for( int i = 0; i< contours.size(); i++ ){
        //std::cout << "contour size is " << contours[i].size() << " index is " << i << std::endl;
        contour_ = contours[i];
        cv::Scalar color = cv::Scalar(0,0,255);
        
        for(int j = 0; j < contour_.size(); j++){
            //image, center of circle, radius, clolr, thickness
            cv::circle(image, contour_[j], 10, color, 5, 10, 0);
        }
        //cv::drawContours( image, contours, i, color, 1, 8);
    }
}

// setup camera and put fps
/*
- (void) cameraSetup {
    
    
    float cam_width = 480; float cam_height = 640;
    //float cam_width = 720; float cam_height = 1280;
    
    // setup frame size
    int view_width = self.view.frame.size.width;
    int view_height = (int)(cam_height*self.view.frame.size.width/cam_width);
    int offset = (self.view.frame.size.height - view_height)/2;
    
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, offset, view_width, view_height)];
    
    [self.view addSubview:imageView_];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView_];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30; // Set the frame rate
    self.videoCamera.grayscaleMode = NO; // Get grayscale
    self.videoCamera.rotateVideo = YES;
    
    // choose different camera resolution
    if (cam_width == 480){
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    }
    else if (cam_width == 720){
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1280x720;
    }
    else {
        std::cout << "wrong camera width " << cam_width << "x" << cam_height << std::endl;
        return;
    }
    
    
    // setup fps text
    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
    [fpsView_ setOpaque:false]; // Set to be Opaque
    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
    [self.view addSubview:fpsView_];
    
    
}
*/

- (void) showFPS {
    // Finally estimate the frames per second (FPS)
    int64 next_time = cv::getTickCount(); // Get the next time stamp
    float fps = (float)cv::getTickFrequency()/(next_time - curr_time_); // Estimate the fps
    curr_time_ = next_time; // Update the time
    NSString *fps_NSStr = [NSString stringWithFormat:@"FPS = %2.2f",fps];
    
    // Have to do this so as to communicate with the main thread
    // to update the text display
    dispatch_sync(dispatch_get_main_queue(), ^{
        fpsView_.text = fps_NSStr;
    });
}

- (void)showImage:(UIImage *) image{
    
    int setframeheight= image.size.height * self.view.frame.size.width / image.size.width;
    
    if( setframeheight <= self.view.frame.size.height){
        imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, setframeheight)];
    }
    else{
        
        int setframewidth = image.size.width * self.view.frame.size.height / image.size.height;
        imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, setframewidth, self.view.frame.size.height)];
    }
    
    [self.view addSubview:imageView_];
}

// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

cv::Mat DrawPts(cv::Mat &display_im, arma::fmat &pts, const cv::Scalar &pts_clr)
{
    std::vector<cv::Point2f> cv_pts = Arma2Points2f(pts); // Convert to vector of Point2fs
    for(int i=0; i<cv_pts.size(); i++) {
        cv::circle(display_im, cv_pts[i], 5, pts_clr,5); // Draw the points
    }
    return display_im; // Return the display image
}

// Quick function to convert Armadillo to OpenCV Points
std::vector<cv::Point2f> Arma2Points2f(arma::fmat &pts)
{
    std::vector<cv::Point2f> cv_pts;
    for(int i=0; i<pts.n_cols; i++) {
        cv_pts.push_back(cv::Point2f(pts(0,i), pts(1,i))); // Add points
    }
    return cv_pts; // Return the vector of OpenCV points
}


@end
