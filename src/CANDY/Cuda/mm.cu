//
// Created by tony on 12/06/24.
//
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
// Utility function to check CUDA errors
inline void checkCudaError(cudaError_t result, char const *const func, const char *const file, int64_t const line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error = " << static_cast<int64_t>(result) << " at " <<
              file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
__global__ void matrixMulCUDA(float *a, float *b, float *c, int64_t M, int64_t N, int64_t K) {
  int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < K) {
    float sum = 0.0;
    for (int64_t i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * K + col];
    }
    c[row * K + col] = sum;
  }
}

torch::Tensor CudaMM(torch::Tensor a, torch::Tensor b) {
  // Ensure input tensors are on the GPU and are contiguous
  a = a.to(at::kCUDA).contiguous();
  b = b.to(at::kCUDA).contiguous();

  // Dimensions
  int64_t M = a.size(0);
  int64_t N = a.size(1);
  int64_t K = b.size(1);

  // Create the output tensor on the GPU

  torch::Tensor c = torch::zeros({M, K}).to(at::kCUDA);

  // Define block and grid sizes
  dim3 threadsPerBlock(4, 4);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the kernel
  matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(a.data_ptr<float>(),
                                                    b.data_ptr<float>(),
                                                    c.data_ptr<float>(),
                                                    M,
                                                    N,
                                                    K);

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Transfer the result tensor to the CPU
  return c.to(torch::kCPU);
}