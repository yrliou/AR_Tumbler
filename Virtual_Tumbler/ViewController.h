//
//  ViewController.h
//  Virtual_Tumbler
//
//  Created by Sam on 2017/4/7.
//  Copyright © 2017年 Sam. All rights reserved.
//

#import <UIKit/UIKit.h>
#include <opencv2/opencv.hpp>
#import <opencv2/highgui/ios.h>


@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    CvVideoCamera* videoCamera;
}
// Declare internal property of videoCamera
@property (nonatomic, retain) CvVideoCamera *videoCamera;

@end

