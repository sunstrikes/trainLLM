#pragma once
#include <cuda_runtime.h>
#include "util.h"
namespace train_llm {

__global__ void convolution2DKernel(
        const float* input,
        const float* kernel,
        float* output,
        int inputWidth,
        int inputHeight,
        int kernelWidth,
        int kernelHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < inputWidth && row < inputHeight) {
        int halfKernelWidth = kernelWidth / 2;
        int halfKernelHeight = kernelHeight / 2;

        float result = 0.0f;

        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row - halfKernelHeight + i;
                int inputCol = col - halfKernelWidth + j;

                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                    result += input[inputRow * inputWidth + inputCol] * kernel[i * kernelWidth + j];
                }
            }
        }

        output[row * inputWidth + col] = result;
    }
}
}  // namespace train_llm
