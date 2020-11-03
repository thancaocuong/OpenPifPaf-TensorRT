#ifndef _POSE_PLUGIN_LIB_H_
#define _POSE_PLUGIN_LIB_H_


// #include <glib.h>
#include "openpifpaf/trt_utils.hpp"
#include "openpifpaf/pifpafnet.hpp"
#include "openpifpaf/npp_preprocess.hpp"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PosePluginCtx PosePluginCtx;
typedef struct PosePluginOutput PosePluginOutput;
// Init parameters structure as input, required for instantiating yoloplugin_lib
typedef struct
{
    // Width at which frame/object will be scaled
    int processingWidth;
    // height at which frame/object will be scaled
    int processingHeight;
    // Flag to indicate whether operating on crops of full frame
    int fullFrame;
    // Plugin config file
    std::string configFilePath;
    std::string engineFilePath;
} PosePluginInitParams;

struct PosePluginCtx
{
    PosePluginInitParams initParams;
    NetworkInfo networkInfo;
    TensorRTNet* inferenceEngine;
    // perf vars
    float inferTime = 0.0, preTime = 0.0, postTime = 0.0;
    float currentInferTime = 0.0, currentPreTime = 0.0, currentPostTime = 0.0;
    uint batchCount = 0, batchSize = 0;
};

// Initialize library context
PosePluginCtx* PosePluginCtxInit(PosePluginInitParams* initParams, size_t batchSize);

// Dequeue processed output
std::vector<InferResult> PosePluginProcess(PosePluginCtx* ctx, std::vector<cv::Mat>& cvmats);
// print performance
void PosePluginPrintPerfInfo(PosePluginCtx* ctx);

// Deinitialize library context
void PosePluginCtxDeinit(PosePluginCtx* ctx);

#ifdef __cplusplus
}

#endif

#endif
