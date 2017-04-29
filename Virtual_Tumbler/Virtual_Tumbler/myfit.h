//
//  myfit.h
//  Estimate_Homography
//
//  Created by Simon Lucey on 9/21/15.
//  Copyright (c) 2015 CMU_16432. All rights reserved.
//

#ifndef __Estimate_Homography__myfit__
#define __Estimate_Homography__myfit__

#define NUMBER_OF_RANSAC_ITERATIONS 30000
#define RANSAC_THRESHOLD 15

#include <stdio.h>
#include "armadillo" // Includes the armadillo library
#include <opencv2/opencv.hpp> // Includes the opencv library
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <vector>
#include <unordered_set>

// Functions for students to fill-in for Assignment 1
arma::fmat myfit_affine(arma::fmat &X, arma::fmat &W);
arma::fmat myproj_affine(arma::fmat &W, arma::fmat &A);
arma::fmat myfit_homography(arma::fmat &X, arma::fmat &W);
arma::fmat myproj_homography(arma::fmat &W, arma::fmat &H);
arma::fmat my_ransac(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2);
arma::fmat myfit_extrinsic(arma::fmat& K, arma::fmat& H);
arma::fmat myproj_extrinsic(arma::fmat& Intrinsic, arma::fmat& Extrinsic, arma::fmat& W);

#endif /* defined(__Estimate_Homography__myfit__) */
