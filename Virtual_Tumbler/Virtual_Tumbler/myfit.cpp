//
//  myfit.cpp
//  Estimate_Homography
//
//  Created by Simon Lucey on 9/21/15.
//  Copyright (c) 2015 CMU_16432. All rights reserved.
//

#include "myfit.h"

// Use the Armadillo namespace
using namespace arma;

//-----------------------------------------------------------------
// Function to return the affine warp between 3D points on a plane
//
// <in>
// X = concatenated matrix of 2D projected points in the image (2xN)
// W = concatenated matrix of 3D points on the plane (3XN)
//
// <out>
// A = 2x3 matrix of affine parameters
fmat myfit_affine(fmat &X, fmat &W) {

    // Fill in the answer here.....
    fmat A; return A;
}
//-----------------------------------------------------------------
// Function to project points using the affine transform
//
// <in>
// W = concatenated matrix of 3D points on the plane (3XN)
// A = 2x3 matrix of affine parameters
//
// <out>
// X = concatenated matrix of 2D projected points in the image (2xN)
fmat myproj_affine(fmat &W, fmat &A) {

    // Fill in the answer here.....
    fmat X; return X;
}

//-----------------------------------------------------------------
// Function to return the affine warp between 3D points on a plane
//
// <in>
// X = concatenated matrix of 2D projected points in the image (2xN)
// W = concatenated matrix of 3D points on the plane (3XN)
//
// <out>
// H = 3x3 homography matrix
fmat myfit_homography(fmat &X, fmat &W) {
    
    fmat AT(9, W.n_cols * 2);
    for (int i = 0; i < W.n_cols; i++) { // for each 3D point
        float u = W[i * 3];
        float v = W[i * 3 + 1];
        float x = X[i * 2];
        float y = X[i * 2 + 1];
        
        int first_eq_index = i * 2 * 9;
        int second_eq_index = first_eq_index + 9;
        
        AT[first_eq_index] = 0;
        AT[first_eq_index + 1] = 0;
        AT[first_eq_index + 2] = 0;
        AT[first_eq_index + 3] = -1 * u;
        AT[first_eq_index + 4] = -1 * v;
        AT[first_eq_index + 5] = -1;
        AT[first_eq_index + 6] = y * u;
        AT[first_eq_index + 7] = y * v;
        AT[first_eq_index + 8] = y;
        
        AT[second_eq_index] = u;
        AT[second_eq_index + 1] = v;
        AT[second_eq_index + 2] = 1;
        AT[second_eq_index + 3] = 0;
        AT[second_eq_index + 4] = 0;
        AT[second_eq_index + 5] = 0;
        AT[second_eq_index + 6] = -1 * x * u;
        AT[second_eq_index + 7] = -1 * x * v;
        AT[second_eq_index + 8] = -1 * x;
    }

    fmat A = trans(AT);

    fmat U;
    fvec s;
    fmat V;
    
    svd(U, s, V, A);
    fvec v = V.col(V.n_cols - 1);
    fmat H = trans(reshape(v, 3, 3));
    
    return H;
}

//-----------------------------------------------------------------
// Function to project points using the affine transform
//
// <in>
// W = concatenated matrix of 3D points on the plane (3XN)
// H = 3x3 homography matrix
//
// <out>
// X = concatenated matrix of 2D projected points in the image (2xN)
fmat myproj_homography(fmat &W, fmat &H) {
    // fill 1 to the last row
    W.shed_row(W.n_rows - 1);
    W.insert_rows(W.n_rows, ones<fmat>(1, W.n_cols));

    fmat X = H * W;

    // normalize to x, y, 1 for each column
    for (int i = 0; i < X.n_cols; i++) { // for each point
        float lamda = 1 / X[i * 3 + 2];
        X[i * 3] *= lamda;
        X[i * 3 + 1] *= lamda;
        X[i * 3 + 2] *= lamda;
    }

    // remove the last row of 1 to return a 2xN array
    X.shed_row(X.n_rows - 1);
    return X;
}

fmat my_ransac(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) {
    std::random_device rd;

    fmat best_inliers_W;
    fmat best_inliers_X;
    fmat outputH;

    fmat all_points1(3, matches.size());
    for (int i = 0; i < matches.size(); i++) {
        all_points1[i * 3] = keypoints1[matches[i].queryIdx].pt.x;
        all_points1[i * 3 + 1] = keypoints1[matches[i].queryIdx].pt.y;
        all_points1[i * 3 + 2] = 1;
    }

    for (int i = 0; i < NUMBER_OF_RANSAC_ITERATIONS; i++) {
        fmat inliers_W;
        fmat inliers_X;

        // generate W from randomly select 4 points from keypoints1
        // generate X from corresponding 4 points in keypoints2
        fmat X, W; // 2x4 and 3x4
        
        std::unordered_set<int> rand_indexes;
        while (rand_indexes.size() != 4) {
            rand_indexes.insert(rd() % matches.size());
        }
        
        for (auto it = rand_indexes.begin(); it != rand_indexes.end(); ++it) {
            int rand_index = *it;
            cv::Point2f point1 = keypoints1[matches[rand_index].queryIdx].pt;
            cv::Point2f point2 = keypoints2[matches[rand_index].trainIdx].pt;
            W.insert_cols(W.n_cols, fvec({point1.x, point1.y, 1}));
            X.insert_cols(X.n_cols, fvec({point2.x, point2.y}));
        }

        // estimate homography by randomly selected 4 points and use it to estimate projection points on image2
        fmat H = myfit_homography(X, W);
        fmat estimateMappingPoints = myproj_homography(all_points1, H);

        // for each point in estimateMappingPoints,
        // decide if it is an inlier based on the distance to the corresponding point in keypoint2
        std::vector<double> distances;
        for (int j = 0; j < matches.size(); j++) {
            float est_x2 = estimateMappingPoints[j * 2];
            float est_y2 = estimateMappingPoints[j * 2 + 1];
            float x2 = keypoints2[matches[j].trainIdx].pt.x;
            float y2 = keypoints2[matches[j].trainIdx].pt.y;
            double distance = sqrt(pow(est_x2 - x2, 2) + pow(est_y2 - y2, 2));
            // std::cout << distance << std::endl;
            if (distance < RANSAC_THRESHOLD) {
                distances.push_back(distance); // for debug
                float x1 = keypoints1[matches[j].queryIdx].pt.x;
                float y1 = keypoints1[matches[j].queryIdx].pt.y;

                // candidate for best_W
                fvec col_w({x1, y1, 1});
                inliers_W.insert_cols(inliers_W.n_cols, col_w);
                // candidate for best_X
                fvec col_x({x2, y2});
                inliers_X.insert_cols(inliers_X.n_cols, col_x);
            }
        }

        if (inliers_W.n_cols >= best_inliers_W.n_cols) { // have equal or more inlier points
            /*
            std::cout << "===iteration " << i << " ========" << std::endl;
            for (int k = 0; k < distances.size(); k++) {
                std::cout << distances[k] << std::endl;
            }
            */
            best_inliers_W = inliers_W;
            best_inliers_X = inliers_X;
        }
    }
    
    std::cout << best_inliers_W.n_cols << std::endl;
    outputH = myfit_homography(best_inliers_X, best_inliers_W);
    
    return outputH;
}

fmat myfit_extrinsic(fmat& K, fmat& H) {
    fmat extrinsic;
    
    // H' = (K^-1)H
    fmat H2 = K.i() * H;
    std::cout << "H2 = " << H2 << std::endl;
    
    // compute SVD of first two columns
    fmat first_two_cols = H2;
    first_two_cols.shed_col(first_two_cols.n_cols - 1);
    
    fmat U;
    fvec s;
    fmat V;
    
    svd(U, s, V, first_two_cols);

    // compute rotation matrix
    fmat middle;
    middle << 1 << 0 << endr
           << 0 << 1 << endr
           << 0 << 0;
    
    fmat rotation = U * middle * trans(V);
    rotation.insert_cols(rotation.n_cols, cross(rotation.col(0), rotation.col(1)));
    
    // check the sign of the last column
    if (det(rotation) == -1) {
        fmat negate_last_col;
        negate_last_col << 1 << 0 << 0 << endr
                        << 0 << 1 << 0 << endr
                        << 0 << 0 << -1;

        rotation = rotation * negate_last_col;
    }
    
    std::cout << "rotation" <<std::endl << rotation << std::endl;
    double lamda = 0;
    for (int n = 0; n < 2; n++) {
        for (int m = 0; m < 3; m++) {
            lamda += H2[n * 3 + m] / rotation[n * 3 + m];
            // std::cout << rotation[n * 3 + m] << std::endl;
        }
    }
    lamda /= 6;
    std::cout << "lamda = " << lamda << std::endl;
    
    
    
    fvec translation = H2.col(H2.n_cols - 1) / lamda;
    std::cout << "translation" << std::endl << translation << std::endl;
    
    // extrinsic.insert_cols(extrinsic.n_cols, rotation.col(0));
    std::cout << "extrinsic" << std::endl;
    extrinsic.insert_cols(extrinsic.n_cols, rotation);
    extrinsic.insert_cols(extrinsic.n_cols, translation);
    
    std::cout << extrinsic << std::endl;
    
    return extrinsic;
}

fmat myproj_extrinsic(fmat& Intrinsic, fmat& Extrinsic, fmat& W) {
    fmat H = Intrinsic * Extrinsic;
    W.insert_rows(W.n_rows, ones<fmat>(1, W.n_cols));
    fmat X = myproj_homography(W, H);
    return X;
}
