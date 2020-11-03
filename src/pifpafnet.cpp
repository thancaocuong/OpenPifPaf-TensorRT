#include "openpifpaf/pifpafnet.hpp"


PifPafNet::PifPafNet(const uint batchSize, const NetworkInfo& networkInfo):
    TensorRTNet(batchSize, networkInfo){};

void PifPafNet::nvPreprocess(const std::vector<cv::Mat>& cvmats, int processingWidth, int processingHeight)
        {
            int input_binding_index = getInputBindingIndex();
            float *inputData = (float*)mDeviceBuffers.at(input_binding_index);
            for(size_t i=0; i<cvmats.size(); i++)
            {
                GPUImg _gpu_original_data8u;
                _gpu_original_data8u.width = cvmats[i].cols;
                _gpu_original_data8u.height = cvmats[i].rows;
                NV_CUDA_CHECK(cudaMalloc(&_gpu_original_data8u.data, cvmats[i].cols*cvmats[i].rows*3));
                cv::Mat rgb_img;
                cv::cvtColor(cvmats[i], rgb_img, cv::COLOR_BGR2RGB);
                cudaMemcpy(_gpu_original_data8u.data,
                           rgb_img.data, rgb_img.cols * rgb_img.rows * 3,
                           cudaMemcpyHostToDevice);
                cudaTensorNormMeanRGB(_gpu_original_data8u.data, _gpu_original_data8u.width, _gpu_original_data8u.height,
                                        inputData, mInputW, mInputH,
                                        make_float2(0.0f, 1.0f),
                                        make_float3(0.485f, 0.456f, 0.406f),
                                        make_float3(0.229f, 0.224f, 0.225f), NULL);
                inputData += mInputSize;
                cudaFree(_gpu_original_data8u.data);
            }
            cudaDeviceSynchronize();
        }

std::vector<InferResult> PifPafNet::decodeBatchOfFrames(const int imageH,const int imageW)
{

    auto pif = mOutputTensors.at(0).hostBuffer;
    auto paf = mOutputTensors.at(1).hostBuffer;
    // TODO decode pif and paf
    std::vector<InferResult> result;
    return result;
}
