#ifndef _CV_MAT_UTILS_HPP_
#define _CV_MAT_UTILS_HPP_

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// convert cv::Mat to string (RGB or BGR image)
cv::Mat cvMatFromString(const std::string image_str, int width, int height);

// convert std::string to cv::Mat (RGB or BGR format for cv::Mat)
std::string cvMatToString(const cv::Mat& cvMat);

std::string cvMatToString_v2(const cv::Mat& cvMat);

cv::Mat cvMatFromString_v2(const std::string image_str, int width, int height);

#endif