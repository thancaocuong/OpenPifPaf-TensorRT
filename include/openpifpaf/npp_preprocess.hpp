#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <npp.h>
#include <opencv2/opencv.hpp>
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight);

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);

void imageResize_32f_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight);

void imageNormalization(void *ptr, int width, int height, cudaStream_t stream);

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);

void cudaTensorNormMeanRGB( void* input, size_t inputWidth, size_t inputHeight,
						     float* output, size_t outputWidth, size_t outputHeight, 
						     const float2& range, const float3& mean, const float3& stdDev,
						     cudaStream_t stream );

void cudaTensorNormMeanBGR( void* input, size_t inputWidth, size_t inputHeight,
						     float* output, size_t outputWidth, size_t outputHeight, 
						     const float2& range, const float3& mean, const float3& stdDev,
						     cudaStream_t stream );