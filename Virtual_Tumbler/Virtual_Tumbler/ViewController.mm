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
#import <AVFoundation/AVFoundation.h>
// #include "armadillo"

@interface ViewController () <AVCaptureVideoDataOutputSampleBufferDelegate> {
    // show FPS
    UITextView *fpsView_;
    int64 curr_time_;
    
    // show image
    UIImageView *imageView_;
    
    // draw layer
    UIImageView *drawView_;
    
    // card tracking and recognition
    int card_recognition; // count for going to card recognition
    cv::vector<cv::vector<cv::Point>> card_corners;
    cv::Mat prevImage; // store the previous image for calculating homography
    float TRACK_RESCALE;
    
    // Used for homography
    cv::BRISK *brisk_detector_;
}

// AVFoundation video
@property AVCaptureDevice* videoDevice;
@property AVCaptureSession* captureSession;
@property AVCaptureVideoDataOutput* captureOutput;
@property dispatch_queue_t captureSessionQueue;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Choose which mode
    int VideoStream = 0; // Video
    //int VideoStream = 1; // card Tracking
    //int VideoStream = 2; // card Recognition
    
    TRACK_RESCALE = 0.50;
    
    //setup Brisk descriptor and detector
    int thresh = 30;
    int octaves = 3;
    float patternSacle = 1.0f;
    brisk_detector_ = new cv::BRISK(thresh, octaves, patternSacle);
    
    [self VideoStillImage:VideoStream];
}

- (void) VideoStillImage:(int) VideoStream{
    
    if (VideoStream == 0){
        [self cameraSetup];
        card_recognition = 0;
        [self.captureSession startRunning];
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
        
        [self plotCircle:processImage points:card_corners];
        cv::cvtColor(processImage, processImage, CV_BGR2RGBA);
        
        // show on the screen
        imageView_.image = [self UIImageFromCVMat:processImage];
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    CVImageBufferRef pixelBuffer =
        CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress( pixelBuffer, 0 );

    // check format of frame buffer
    if (CVPixelBufferGetPixelFormatType(pixelBuffer) != kCVPixelFormatType_32BGRA) {
        std::cout << "Err: not expected pixel format" << std::endl;
        CVPixelBufferUnlockBaseAddress( pixelBuffer, 0 );
        return;
    }

    int format_opencv = CV_8UC4;
    void* bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t bufferWidth = CVPixelBufferGetWidth(pixelBuffer);
    size_t bufferHeight = CVPixelBufferGetHeight(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    
    CGColorSpaceRef colorSpace;
    CGContextRef context;

    // NSLog(@"%zi, %zi, %zi", bufferWidth,bufferHeight,bytesPerRow);

    // put frame buffer in open cv Mat, no memory copied
    cv::Mat bufImg(bufferHeight, bufferWidth, format_opencv, bufferAddress, bytesPerRow);

    // start processing the frame image
    cv::Mat image = bufImg.clone();
    // change to BGR
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

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

    // show result image
    CGImageRef dstImage;
    CGBitmapInfo bitmapInfo;

    if (image.channels() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
        bitmapInfo = kCGImageAlphaNone;
    } else if (image.channels() == 3) {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        bitmapInfo = kCGImageAlphaNone;
        bitmapInfo |= kCGBitmapByteOrder32Big;
        
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        bitmapInfo = kCGImageAlphaPremultipliedFirst;
        bitmapInfo |= kCGBitmapByteOrder32Big;
        
    }
    
    NSData *data = [NSData dataWithBytes:image.data length:image.elemSize()*image.total()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
        
    // Creating CGImage from cv::Mat
    dstImage = CGImageCreate(image.cols,                                 // width
                             image.rows,                                 // height
                             8,                                          // bits per component
                             8 * image.elemSize(),                       // bits per pixel
                             image.step,                                 // bytesPerRow
                             colorSpace,                                 // colorspace
                             bitmapInfo,                                 // bitmap info
                             provider,                                   // CGDataProviderRef
                             NULL,                                       // decode
                             false,                                      // should interpolate
                             kCGRenderingIntentDefault                   // intent
                             );
        
    CGDataProviderRelease(provider);

    UIImage *show_image = [UIImage imageWithCGImage:dstImage];
    
    // Have to do this so as to communicate with the main thread
    // to update the image display
    dispatch_sync(dispatch_get_main_queue(), ^{
        drawView_.image = show_image;
    });
    
    //End processing
    CGImageRelease(dstImage);
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress( pixelBuffer, 0 );
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
    
    self.captureSession = [[AVCaptureSession alloc] init];
    AVCaptureVideoPreviewLayer *previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    
    self.videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *captureInput = [AVCaptureDeviceInput deviceInputWithDevice:self.videoDevice error:nil];

    self.captureOutput = [[AVCaptureVideoDataOutput alloc] init];
    self.captureOutput.alwaysDiscardsLateVideoFrames = true;
    self.captureSessionQueue = dispatch_queue_create("capture_session_queue", NULL);
    [self.captureOutput setSampleBufferDelegate:self queue:self.captureSessionQueue];
    
    // Define the pixel format for the video data output
    NSString * key = (NSString*)kCVPixelBufferPixelFormatTypeKey;
    NSNumber * value = [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA];
    NSDictionary * settings = @{key:value};
    [self.captureOutput setVideoSettings:settings];
    
    [self.captureSession addInput:captureInput];
    [self.captureSession addOutput:self.captureOutput];

    // float cam_width = 1080; float cam_height = 1920;
    float cam_width = 480; float cam_height = 640;
    // float cam_width = 720; float cam_height = 1280;
    
    if (cam_height == 640 && cam_width == 480 &&
        [self.captureSession canSetSessionPreset:AVCaptureSessionPreset640x480]) {
        [self.captureSession setSessionPreset:AVCaptureSessionPreset640x480];
    } else if (cam_height == 1280 && cam_width == 720 &&
        [self.captureSession canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
        [self.captureSession setSessionPreset:AVCaptureSessionPreset1280x720];
    }
    
    // setup frame size
    int view_width = self.view.frame.size.width;
    int view_height = (int)(cam_height*self.view.frame.size.width/cam_width);
    int offset = (self.view.frame.size.height - view_height)/2;
    
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, offset, view_width, view_height)];

    previewLayer.frame = imageView_.bounds;
    // [imageView_.layer addSublayer:previewLayer];
    
    [self.view addSubview:imageView_];
    [self.view.layer addSublayer:previewLayer];
    
    // setup fps text
    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
    [fpsView_ setOpaque:false]; // Set to be Opaque
    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
    [self.view addSubview:fpsView_];
    
    // setup draw corner layer
    drawView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, offset, view_width, view_height)];
    [drawView_ setOpaque:false];
    [drawView_ setBackgroundColor:[UIColor clearColor]];
    [self.view addSubview:drawView_];
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
