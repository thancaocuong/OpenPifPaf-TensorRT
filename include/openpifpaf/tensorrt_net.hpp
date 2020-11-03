
#ifndef _BASE_TENSORRT_NET_HPP_
#define _BASE_TENSORRT_NET_HPP_

#include <vector>
#include <iostream>
#include <fstream>

#include "NvInfer.h"

#include "openpifpaf/trt_utils.hpp"
#include "openpifpaf/human.hpp"
#include "openpifpaf/json.hpp"
#include "openpifpaf/npp_preprocess.hpp"


class TensorRTNet
{
public:
    int getInputH() const { return mInputH; }
    int getInputW() const { return mInputW; }
    int getInputC() const { return mInputC; }
    int getOutputH() const { return mOutputH; }
    int getOutputW() const { return mOutputW; }
    uint64_t getInputSize() const { return mInputSize; };
    int getInputBindingIndex() const { return mInputBindingIndex; };
    bool isPrintPerfInfo() const { return mPrintPerfInfo; }
    float* getInputBuffer() { return (float*)mDeviceBuffers.at(mInputBindingIndex);};
    /*
    * Infer functions
    */
    void doInference();
    virtual void nvPreprocess(const std::vector<cv::Mat>& cvmats, int processingWidth, int processingHeight) = 0;
    virtual std::vector<InferResult> decodeBatchOfFrames(const int imageH, const int imageW) = 0;

    virtual ~TensorRTNet();

protected:
    TensorRTNet(const uint batchSize, const NetworkInfo& networkInfo);
    std::string mEnginePath;
    std::string mConfigFilePath;
    int mInputH;
    int mInputW;
    int mInputC;
    int mOutputH;
    int mOutputW;
    uint64_t mInputSize;
    bool mPrintPerfInfo;
    cudaStream_t mStream;
    Logger mLogger;
    // TRT specific members
    uint mBatchSize;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;
    nvinfer1::IRuntime* mTrtRunTime;
    std::vector<void*> mDeviceBuffers;
    int mInputBindingIndex;
    std::vector<int> mOutputBindingIndexes;
    std::vector<TensorInfo> mOutputTensors;
    int mNumberInput=0;

    typedef struct GPUImg {
    void *data;
    int width;
    int height;
    int channel;
    } GPUImg;

private:
    void loadEngine();
    void parseConfigFile();
    void allocateBuffers();
    bool verifyTensorRTNetEngine();
    void getBindingIndxes();
};

#endif // _BASE_TENSORRT_NET_HPP_
