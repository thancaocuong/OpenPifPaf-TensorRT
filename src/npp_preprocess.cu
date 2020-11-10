#include "openpifpaf/npp_preprocess.hpp"

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(uchar);

    NppiRect oSrcROI;
    oSrcROI.x = imgROI.x;
    oSrcROI.y = imgROI.y;
    oSrcROI.width = imgROI.width;
    oSrcROI.height = imgROI.height;

    int nDstStep = dstWidth * 3 * sizeof(uchar);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = 1.0 * dstWidth / oSrcROI.width;
    double nYFactor = 1.0 * dstHeight / oSrcROI.height;
    double nXShift = - oSrcROI.x * nXFactor ;
    double nYShift = - oSrcROI.y * nYFactor;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_8u_C3R((const Npp8u *)src, oSrcSize, nSrcStep, oSrcROI, (Npp8u *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );

    if(ret != NPP_SUCCESS) {
        printf("imageROIResize8U3C failed %d.\n", ret);
    }
}

__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.z, color.y, color.x);
}

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    convertBGR2RGBfloatKernel<<<grids, blocks>>>((uchar3 *)src, (float3 *)dst, width, height);
}


// ---------------------
void imageResize_32f_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(float);

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3 * sizeof(float);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);
    double nXShift = 0;
    double nYShift = 0;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_32f_C3R((const Npp32f *)src, oSrcSize, nSrcStep, oSrcROI, (Npp32f *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        printf("imageResize_32f_C3R failed %d.\n", ret);
    }
}

// ---------------------
// #### NORMALIZATION ####
// ####      RGB      ####
// ---------------------
__global__ void imageNormalizationKernel(float3 *ptr, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];
    color.x = (color.x - 255.0*0.485)/(255.0*0.229);
    color.y = (color.y - 255.0*0.456)/(255.0*0.224);
    color.z = (color.z - 255.0*0.406)/(255.0*0.225);
    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalization(void *ptr, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationKernel<<<grids, blocks>>>((float3 *)ptr, width, height);
}
// ---------------------
// #### SPLIT ####
// ---------------------
__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
    dst[y * width + x + width * height] = color.y;
    dst[y * width + x + width * height * 2] = color.z;
}

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    imageSplitKernel<<<grids, blocks>>>((float3 *)src, (float *)dst, width, height);
}

template<typename T, bool isBGR>
__global__ void gpuTensorNormMean( T* input, int iWidth, float* output, int oWidth,
                                    int oHeight, float2 scale, float multiplier, float min_value,
                                    const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];
	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
                             : make_float3(px.x, px.y, px.z);

	output[n * 0 + m] = ((rgb.x * multiplier + min_value) - mean.x) / stdDev.x;
	output[n * 1 + m] = ((rgb.y * multiplier + min_value) - mean.y) / stdDev.y;
	output[n * 2 + m] = ((rgb.z * multiplier + min_value) - mean.z) / stdDev.z;
}

template<bool isBGR>
void launchTensorNormMean( void* input, size_t inputWidth, size_t inputHeight,
						    float* output, size_t outputWidth, size_t outputHeight, 
						    const float2& range, const float3& mean, const float3& stdDev,
						    cudaStream_t stream )
{
	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuTensorNormMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar3*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
}

// cudaTensorNormMeanRGB
void cudaTensorNormMeanRGB( void* input, size_t inputWidth, size_t inputHeight,
						     float* output, size_t outputWidth, size_t outputHeight, 
						     const float2& range, const float3& mean, const float3& stdDev,
						     cudaStream_t stream )
{
	launchTensorNormMean<false>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, range, mean, stdDev, stream);
}
// cudaTensorNormMeanBGR
void cudaTensorNormMeanBGR( void* input, size_t inputWidth, size_t inputHeight,
                                    float* output, size_t outputWidth, size_t outputHeight, 
                                    const float2& range, const float3& mean, const float3& stdDev,
                                    cudaStream_t stream )
{
    launchTensorNormMean<true>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, range, mean, stdDev, stream);
}
