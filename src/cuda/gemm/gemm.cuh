#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.h>

#include "kernel_0.cuh"
#include "kernel_1.cuh"
#include "kernel_2.cuh"
#include "kernel_3.cuh"
#include "kernel_4.cuh"

void launch_kernel_cublas(int M, int N, int K, float alpha, float* A, float* B,
                          float beta, float* C, cublasHandle_t handle) {
  cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N);
}
