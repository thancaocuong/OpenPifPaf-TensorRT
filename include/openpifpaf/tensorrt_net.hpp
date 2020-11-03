
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

#if NV_TENSORRT_MAJOR > 5
typedef nvinfer1::Dims3 Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

#elif NV_TENSORRT_MAJOR > 1
typedef nvinfer1::DimsCHW Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

#else
typedef nvinfer1::Dims3 Dims3;

#define DIMS_C(x) x.c
#define DIMS_H(x) x.h
#define DIMS_W(x) x.w

#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR 1
#define NV_TENSORRT_MINOR 0
#endif
#endif

enum modelType
{
	MODEL_CUSTOM = 0,	/**< Created directly with TensorRT API */
	MODEL_ONNX,		/**< ONNX */
	MODEL_ENGINE		/**< TensorRT engine/plan */
};

modelType modelTypeFromStr( const char* str );

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
    modelType mModelType;

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
