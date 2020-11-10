#pragma once
#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas.h>
#include <iostream>

void upsampleGpu(const float* input, float* outputint, int N, int C, int H , int W, int scale, int threadCount);
