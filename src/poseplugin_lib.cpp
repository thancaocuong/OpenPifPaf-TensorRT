#include "openpifpaf/poseplugin_lib.hpp"
#include "openpifpaf/tensorrt_net.hpp"
#include "openpifpaf/human.hpp"
#include <iomanip>
#include <sys/time.h>


static void decodeBatchPoseEstimation(const PosePluginCtx* ctx, std::vector<InferResult>& outputs)
{
    auto inferResultVector = ctx->inferenceEngine->decodeBatchOfFrames(
                                                                       ctx->initParams.processingHeight,
                                                                       ctx->initParams.processingWidth);
    for (auto it = inferResultVector.begin(); it != inferResultVector.end(); it++)
        outputs.push_back(*it);
}


PosePluginCtx* PosePluginCtxInit(PosePluginInitParams* initParams, size_t batchSize)
{
    PosePluginCtx* ctx = new PosePluginCtx;
    ctx->initParams = *initParams;
    ctx->batchSize = batchSize;
    ctx->networkInfo = NetworkInfo{ctx->initParams.configFilePath, ctx->initParams.engineFilePath};
    ctx->inferenceEngine = new PifPafNet(batchSize, ctx->networkInfo);
    return ctx;
}

std::vector<InferResult> PosePluginProcess(PosePluginCtx* ctx, std::vector<cv::Mat>& cvmats)
{
    std::vector<InferResult> outputs;
    cv::Mat preprocessedImages;
    struct timeval preStart, preEnd, inferStart, inferEnd, postStart, postEnd;
    double preElapsed = 0.0, inferElapsed = 0.0, postElapsed = 0.0;
    if (cvmats.size() > 0)
    {
        gettimeofday(&preStart, NULL);
        ctx->inferenceEngine->nvPreprocess(cvmats,
                                            ctx->initParams.processingWidth,
                                            ctx->initParams.processingHeight);
        gettimeofday(&preEnd, NULL);

        gettimeofday(&inferStart, NULL);
        ctx->inferenceEngine->doInference();
        gettimeofday(&inferEnd, NULL);

        gettimeofday(&postStart, NULL);
        decodeBatchPoseEstimation(ctx, outputs);
        gettimeofday(&postEnd, NULL);
    }
    if (ctx->inferenceEngine->isPrintPerfInfo())
    {
        preElapsed
            = ((preEnd.tv_sec - preStart.tv_sec) + (preEnd.tv_usec - preStart.tv_usec) / 1000000.0)
            * (1000/ctx->batchSize);
        inferElapsed = ((inferEnd.tv_sec - inferStart.tv_sec)
                        + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * (1000/ctx->batchSize);
        postElapsed = ((postEnd.tv_sec - postStart.tv_sec)
                       + (postEnd.tv_usec - postStart.tv_usec) / 1000000.0)
            * (1000/ctx->batchSize);

        ctx->currentInferTime = inferElapsed;
        ctx->currentPreTime = preElapsed;
        ctx->currentPostTime = postElapsed;

        ctx->inferTime += inferElapsed;
        ctx->preTime += preElapsed;
        ctx->postTime += postElapsed;
        ++ctx->batchCount;
    }
    return outputs;
}

void PosePluginPrintPerfInfo(PosePluginCtx* ctx)
{
    if (ctx->inferenceEngine->isPrintPerfInfo())
    {
        std::cout << "Batch Size : " << ctx->batchSize;
        std::cout << std::fixed << std::setprecision(4)
                  << " PreProcess : " << ctx->preTime / ctx->batchCount
                  << " ms Inference : " << ctx->inferTime / ctx->batchCount
                  << " ms PostProcess : " << ctx->postTime / ctx->batchCount << " ms Total : "
                  << (ctx->preTime + ctx->postTime + ctx->inferTime) / ctx->batchCount
                  << " ms per Image" << std::endl;
    }
}

void PosePluginCtxDeinit(PosePluginCtx* ctx)
{
    std::cout << "Pose Estimation Plugin Perf Summary " << std::endl;
    PosePluginPrintPerfInfo(ctx);

    delete ctx->inferenceEngine;
    delete ctx;
}
