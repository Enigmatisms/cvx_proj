#pragma once
#include <vector>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>

/**
 * If our experiment in python is okay, then we start a C++ version for faster computation
 */

class Matcher {
public:
    Matcher(const cv::Mat& c_img, const cv::Mat& s_img):
        center_img(c_img), source_image(s_img)
    {   
        ;
    }

    ~Matcher() = default;

    void coarse_matching(const std::vector<cv::KeyPoint>& c_kpts, const std::vector<cv::KeyPoint>& s_kpts);

private:
    const cv::Mat& center_img, source_image;
};