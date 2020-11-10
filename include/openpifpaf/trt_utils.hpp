
/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

/* OpenCV headers */
// #include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <set>
#include <vector>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sys/stat.h>
#include <memory>
#include <algorithm>
#include <cudnn.h>
#include <stdint.h>
#include <string>
#include "NvInfer.h"
#include "openpifpaf/json.hpp"
using namespace std;

/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
    std::string configFilePath;
    std::string enginePath;
};

/**
 * Holds information about an output tensor of the Pose network.
 */
struct TensorInfo
{
    uint64_t volume{0};
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};

#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

class Profiler : public nvinfer1::IProfiler
    {
    public:
        void printLayerTimes(int itrationsTimes)
        {
            float totalTime = 0;
            for (size_t i = 0; i < mProfile.size(); i++)
            {
                printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / itrationsTimes);
                totalTime += mProfile[i].second;
            }
            printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
        }
    private:
        typedef std::pair<std::string, float> Record;
        std::vector<Record> mProfile;

        virtual void reportLayerTime(const char* layerName, float ms)
        {
            auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
            if (record == mProfile.end())
                mProfile.push_back(std::make_pair(layerName, ms));
            else
                record->second += ms;
        }
    };


class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

// Common helper functions
// cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages, const int& inputH,
//                          const int& inputW);
std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName);
std::vector<std::string> loadListFromTextFile(const std::string filename);

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, Logger& logger);
std::string dimsToString(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims);
nlohmann::json readJsonFile(const std::string file_name);
int* topologyFromJson(const nlohmann::json json, const int pafChannels);
vector<vector<int>> skeletonFromJson(const nlohmann::json json, const int pafChannels);
void serialize(const string outFileName, float* arr, int rows, int cols);
#endif
