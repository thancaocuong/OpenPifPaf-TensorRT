#include "openpifpaf/cv_mat_utils.hpp"

cv::Mat cvMatFromString(const std::string image_str, int width, int height)
{
    std::vector<uchar> data(image_str.begin(), image_str.end());
    // cv::Mat cv_image(height, width, CV_8UC3, const_cast<char*>(image_str.c_str()));
    cv::Mat cv_image  = cv::imdecode(data, 1);
    return cv_image.clone();
}

std::string cvMatToString(const cv::Mat& cvMat)
{
    std::vector<uchar> buf;
    cv::imencode(".jpg", cvMat, buf);
    std::string cvMatAsString(buf.begin(), buf.end());
    return cvMatAsString;
}

std::string cvMatToString_v2(const cv::Mat& cvMat)
{
    int dataSize = cvMat.total() * cvMat.elemSize();
    //convert to bytes
    std::vector<char> vec(dataSize);
    memcpy(&vec[0], reinterpret_cast<char *>(cvMat.data), dataSize);
    std::string img_str(vec.begin(), vec.end());
    return img_str;
}

cv::Mat cvMatFromString_v2(const std::string image_str, int width, int height){

    cv::Mat cv_image(height, width, CV_8UC3, const_cast<char*>(image_str.c_str()));
    return cv_image.clone();
}