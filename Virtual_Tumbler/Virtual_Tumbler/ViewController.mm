//
//  ViewController.m
//  Virtual_Tumbler
//
//  Created by Sam on 2017/4/7.
//  Copyright © 2017年 Sam. All rights reserved.
//

#import "ViewController.h"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "cardRecognition.h"
#include "cardTracking.h"


@interface ViewController (){
    UIImageView *imageView_;
    UITextView *fpsView_;
    int64 curr_time_;
    int card_recognition;
    cv::vector<cv::vector<cv::Point>> card_corners;
    
    cv::Mat prevImage;
    //cv::vector<cv::Point2f> prefeaturesCorners;
    float TRACK_RESCALE;
    
    // SURF is too slow
    //cv::SurfFeatureDetector *surfDetector;
    //cv::SurfDescriptorExtractor *surfExtractor;
    //cv::Ptr<cv::FeatureDetector> Detector_;
    //cv::Ptr<cv::DescriptorExtractor> Extractor_;
    cv::BRISK *brisk_detector_;
    
}

@end

@implementation ViewController

@synthesize videoCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    Boolean VideoStream = true;
    //Boolean VideoStream = false;
    
    TRACK_RESCALE = 0.60;
    
    //int minHessian = 800;
    //surfDetector = new cv::SurfFeatureDetector(minHessian); // set the detector
    //surfExtractor = new cv::SurfDescriptorExtractor(); // Set the extractor
    int thresh = 30;
    int octaves = 3;
    float patternSacle = 1.0f;
    brisk_detector_ = new cv::BRISK(thresh, octaves, patternSacle);
    
    
    [self VideoStillImage:VideoStream];
}

- (void) VideoStillImage:(Boolean) VideoStream{
    
    if (VideoStream){
        [self cameraSetup];
        card_recognition = 0;
        [videoCamera start];
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
        
        //UIImage *inputImage = image_4cards;
        //UIImage *inputImage = image_3cards_hand;
        //UIImage *inputImage = image_3cards_half;
        //UIImage *inputImage = image_test_3cards;
        //UIImage *inputImage = image_test_3cards_dark;
        UIImage *inputImage = image_test_1card;
        
        // set the ImageView_ for still image
        [self showImage:inputImage];
        
        cv::Mat cvImage = [self cvMatFromUIImage:inputImage];
        cv::Mat processImage;
        cv::cvtColor(cvImage, processImage, CV_RGBA2BGR);
        
        // process image
        cv::vector<cv::vector<cv::Point>> card_corners;
        card_corners = cardRecognition(processImage);
        
        // Test fillpoly to get a mask
        cv::fillConvexPoly(processImage, card_corners[0], cv::Scalar(0,255,0));
        
        [self plotCircle:processImage points:card_corners];
        cv::cvtColor(processImage, processImage, CV_BGR2RGBA);
        
        
        // show on the screen
        imageView_.image = [self UIImageFromCVMat:processImage];
        
        //imageView_.image = inputImage;
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

// Function to run apply image on
- (void) processImage:(cv::Mat &)image
{
    
    card_corners = cardRecognition(image);
    [self plotCircle:image points:card_corners];
    
    
    /*
    // blur image before downsampling
    cv::Mat grayImage;
    cv::Mat resizeImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImage, resizeImage, cv::Size(5,5), 1.0, 1.0);
    cv::resize(resizeImage, grayImage, cv::Size(), TRACK_RESCALE, TRACK_RESCALE);
    //cv::cvtColor(resizeImage, grayImage, cv::COLOR_BGR2GRAY);
    
    
    if(card_recognition < 60){
        card_corners = cardRecognition(image);
        prevImage = grayImage.clone();
        
        //featureTrack(grayImage, prefeaturesCorners);
        
        //plot circle
        [self plotCircle:image points:card_corners];
        
        card_recognition += 1;
    }
    else{
        // not yet finish can move card independently
        //cardTracking(card_corners, grayImage, prevImage, prefeaturesCorners);
        
        // card can't move independently, only move camera
        cardAllFindhomography(prevImage, grayImage, card_corners, TRACK_RESCALE, brisk_detector_);
        //cardAllFindhomography(prevImage, grayImage, card_corners, TRACK_RESCALE, surfExtractor, surfDetector);
        
        trackingCorner(grayImage, card_corners, image, TRACK_RESCALE);
        
        prevImage = grayImage.clone();
        [self plotCircle:image points:card_corners];
    }
    */
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

@end
