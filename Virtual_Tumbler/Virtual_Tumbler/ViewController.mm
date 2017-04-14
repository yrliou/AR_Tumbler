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


@interface ViewController (){
    UIImageView *imageView_;
    UITextView *fpsView_;
    int64 curr_time_;
}

@end

@implementation ViewController

@synthesize videoCamera;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    Boolean VideoStream = true;
    //Boolean VideoStream = false;
    [self VideoStillImage:VideoStream];
}

- (void) VideoStillImage:(Boolean) VideoStream{
    
    if (VideoStream){
        [self cameraSetup];
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
        
        // read image_test_many image
        UIImage *image_test_3cards_dark = [UIImage imageNamed:@"image_test_3cards_dark.jpg"];
        if(image_test_many == nil) std::cout << "Cannot read in the file image_test_3cards_dark.jpg!!" << std::endl;
        
        //UIImage *inputImage = image_test_3cards;
        //UIImage *inputImage = image_test_3cards_dark;
        UIImage *inputImage = image_test_1card;
        
        // set the ImageView_ for still image
        [self showImage:inputImage];
        
        cv::Mat cvImage = [self cvMatFromUIImage:inputImage];
        cv::Mat processImage;
        cv::cvtColor(cvImage, processImage, CV_RGBA2BGR);
        
        // process image
        
        colorFilter(processImage);
        
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
    // show FPS
    colorFilter(image);
    [self showFPS];
}


// setup camera and put fps
- (void) cameraSetup {
    
    
    //float cam_width = 480; float cam_height = 640;
    float cam_width = 720; float cam_height = 1280;
    
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
