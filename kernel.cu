// Gaussian Blur using CUDA
//  - Author: Tomash Mikulevich
//  - Created using: Microsoft Visual Studio Community 2022 (64-bit) 17.4.2
//    - CPU: AMD Ryzen 7 5800H | GPU: NVIDIA GeForce RTX 3050 Ti Laptop
//  - Last edited: 06.02.2023

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define THREADS_NUM 32

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector_types.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stb_image.h"
#include "stb_image_write.h"


double calcError(uchar4* result1, uchar4* result2, int height, int width, int channel) {
    double maxDiff = 0;
    double maxVal = 0;

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if ((col + width * row) < width * height) {
                switch (channel) {
                    case 1:
                        maxDiff = fmax(maxDiff, fabs(result2[col + width * row].x - result1[col + width * row].x));
                        maxVal = fmax(maxVal, fabs(result1[col + width * row].x));
                        break;
                    case 2:
                        maxDiff = fmax(maxDiff, fabs(result2[col + width * row].y - result1[col + width * row].y));
                        maxVal = fmax(maxVal, fabs(result1[col + width * row].y));
                        break;
                    case 3:
                        maxDiff = fmax(maxDiff, fabs(result2[col + width * row].z - result1[col + width * row].z));
                        maxVal = fmax(maxVal, fabs(result1[col + width * row].z));
                        break;
                }
            }
        }
    }

    return 100 * maxDiff / maxVal;
}


float calcRating(float time1, float time2) {
    return time1 / time2;
}


void gaussianBlurCPU(uchar4* input, double* filter, int height, int width, int filterWidth, uchar4* output) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if ((col + width * row) < width * height && (row + height * col) < width * height) {
                double outR = 0.0;
                double outG = 0.0;
                double outB = 0.0;

                for (int i = -filterWidth / 2; i <= filterWidth / 2; ++i) {
                    for (int j = -filterWidth / 2; j <= filterWidth / 2; ++j) {
                        int h = min(max(row + i, 0), height - 1);
                        int w = min(max(col + j, 0), width - 1);
                        int ind1 = w + width * h;

                        double pixelOutR = input[ind1].x;
                        double pixelOutG = input[ind1].y;
                        double pixelOutB = input[ind1].z;

                        int ind2 = (i + filterWidth / 2) * filterWidth + j + filterWidth / 2;
                        double weight = filter[ind2];

                        outR += pixelOutR * weight;
                        outG += pixelOutG * weight;
                        outB += pixelOutB * weight;
                    }
                }

                output[col + width * row].x = outR;
                output[col + width * row].y = outG;
                output[col + width * row].z = outB;
                output[col + width * row].w = 255;
            }
        }
    }
}


__global__ void gaussianBlurGPU(uchar4* input, double* filter, int height, int width, int filterWidth, uchar4* output) {
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    double outR = 0.0;
    double outG = 0.0;
    double outB = 0.0;

    if ((col + width * row) < width * height && (row + height * col) < width * height) {
        for (int i = -filterWidth / 2; i <= filterWidth / 2; ++i) {
            for (int j = -filterWidth / 2; j <= filterWidth / 2; ++j) {
                int h = min(max(row + i, 0), height - 1);
                int w = min(max(col + j, 0), width - 1);

                int ind1 = w + width * h;

                double pixelOutR = input[ind1].x;
                double pixelOutG = input[ind1].y;
                double pixelOutB = input[ind1].z;

                int ind2 = (i + filterWidth / 2) * filterWidth + j + filterWidth / 2;
                double weight = filter[ind2];

                outR += pixelOutR * weight;
                outG += pixelOutG * weight;
                outB += pixelOutB * weight;
            }

            output[col + width * row].x = outR;
            output[col + width * row].y = outG;
            output[col + width * row].z = outB;
            output[col + width * row].w = 255;
        }
    }
}


double* createGaussianFilter(int filterWidth, double sigma) {
    double* filterResult = (double*)malloc(filterWidth * filterWidth * sizeof(double));
    double sum = 0.;

    for (int row = -filterWidth / 2; row <= filterWidth / 2; ++row) {
        for (int col = -filterWidth / 2; col <= filterWidth / 2; ++col) {
            double weight = exp(-static_cast<double>(col * col + row * row) / (2.f * sigma * sigma));
            int i = (row + filterWidth / 2) * filterWidth + col + filterWidth / 2;

            filterResult[i] = weight;
            sum += weight;
        }
    }

    for (int row = -filterWidth / 2; row <= filterWidth / 2; ++row) {
        for (int col = -filterWidth / 2; col <= filterWidth / 2; ++col) {
            int i = (row + filterWidth / 2) * filterWidth + col + filterWidth / 2;

            filterResult[i] *= 1.f / sum;
        }
    }

    return filterResult;
}


int main() {

    // -------------- Initializing -------------- 

    char* inputPath;
    char* outputPathCPU;
    char* outputPathGPU;

    inputPath = "input_1.jpg";
    outputPathCPU = "outputCPU_1.jpg";
    outputPathGPU = "outputGPU_1.jpg";

    // inputPath = "input_2.png";
    // outputPathCPU = "outputCPU_2.png";
    // outputPathGPU = "outputGPU_2.png";

    int numChannels = 4;
    int h;
    int w;
    int c;

    int gaussianFilterWidth = 7 * 7;
    double sigma = 3.;

    uint8_t* image = stbi_load(inputPath, &w, &h, &c, numChannels);
    if (image == NULL) {
        printf("Error during loading file! \n");
        return 0;
    }
    else 
        printf("Image is loaded! Wait for results ... \n");

    // -------------- CPU -------------- 

    uchar4* inputImg = (uchar4*)malloc(w * h * sizeof(uchar4));
    uchar4* outputImgCPU = (uchar4*)malloc(w * h * sizeof(uchar4));
    uchar4* outputImgGPU = (uchar4*)malloc(w * h * sizeof(uchar4));
    double* gaussianFilter = createGaussianFilter(gaussianFilterWidth, sigma);

    for (int i = 0; i < numChannels * w * h; i++) {
        switch (i % numChannels) {
            case 0:
                inputImg[i / numChannels].x = image[i];
                break;
            case 1:
                inputImg[i / numChannels].y = image[i];
                break;
            case 2:
                inputImg[i / numChannels].z = image[i];
                break;
            case 3:
                inputImg[i / numChannels].w = image[i];
                break;
        }
    }

    auto begin = std::chrono::high_resolution_clock::now();

    gaussianBlurCPU(inputImg, gaussianFilter, h, w, gaussianFilterWidth, outputImgCPU);

    auto end = std::chrono::high_resolution_clock::now();
    auto timeCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() * 1e-6;

    printf("CPU Done! Time: %.3f ms.\n", timeCPU);

    stbi_write_jpg(outputPathCPU, w, h, numChannels, outputImgCPU, 100);

    // -------------- GPU -------------- 

    uchar4* d_inputImg;
    uchar4* d_outputImg;
    double* d_gaussianFilter;

    cudaMalloc(&d_inputImg, w * h * sizeof(uchar4));
    cudaMalloc(&d_outputImg, w * h * sizeof(uchar4));
    cudaMalloc(&d_gaussianFilter, gaussianFilterWidth * gaussianFilterWidth * sizeof(double));

    cudaMemcpy(d_inputImg, inputImg, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianFilter, gaussianFilter, gaussianFilterWidth * gaussianFilterWidth * sizeof(double), cudaMemcpyHostToDevice);

    int xDim = (w % THREADS_NUM == 0) ? (int)(w / THREADS_NUM) : (int)ceil(w / THREADS_NUM) + 1;
    int yDim = (h % THREADS_NUM == 0) ? (int)(h / THREADS_NUM) : (int)ceil(h / THREADS_NUM) + 1;

    float timeGPU;
    dim3 BLOCKS_GRID(xDim, yDim);
    dim3 THREADS_GRID(THREADS_NUM, THREADS_NUM); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gaussianBlurGPU <<< BLOCKS_GRID, THREADS_GRID >>> (d_inputImg, d_gaussianFilter, h, w, gaussianFilterWidth, d_outputImg);

    cudaEventRecord(stop, 0);   cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGPU, start, stop);
    cudaEventDestroy(start);    cudaEventDestroy(stop);

    cudaMemcpy(outputImgGPU, d_outputImg, (w * h * sizeof(uchar4)), cudaMemcpyDeviceToHost);
    printf("GPU Done! Time: %.3f ms.\n", timeGPU);

    stbi_write_jpg(outputPathGPU, w, h, numChannels, outputImgGPU, 100);

    // -------------- Сomparison -------------- 

    printf("\nError (max pixel difference between CPU and GPU results):");
    printf("\nR: %.2f%%.\n", calcError(outputImgCPU, outputImgGPU, h, w, 1));
    printf("G: %.2f%%.\n", calcError(outputImgCPU, outputImgGPU, h, w, 2));
    printf("B: %.2f%%.\n", calcError(outputImgCPU, outputImgGPU, h, w, 3));
    printf("Time perfomance: x%.3f.\n", calcRating(timeCPU, timeGPU));

    // -------------- Cleanup -------------- 

    free(inputImg);
    free(outputImgCPU);
    free(outputImgGPU);

    cudaFree(d_inputImg);
    cudaFree(d_outputImg);
    cudaFree(d_gaussianFilter);

    return 0;
}
