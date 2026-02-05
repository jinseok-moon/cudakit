#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cudakit {

__global__ void muladd_kernel(int numel, const float *a, const float *b,
                              float c, float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor &a,
                         const at::Tensor &b,
                         double c) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(a.device().type() == at::kCUDA);
  TORCH_CHECK(b.device().type() == at::kCUDA);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = result.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  auto stream = at::cuda::getCurrentCUDAStream();

  muladd_kernel<<<(numel + 255) / 256, 256, 0, stream.stream()>>>(
      numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float *a, const float *b,
                           float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor &a,
                      const at::Tensor &b) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(a.device().type() == at::kCUDA);
  TORCH_CHECK(b.device().type() == at::kCUDA);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = result.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  auto stream = at::cuda::getCurrentCUDAStream();

  mul_kernel<<<(numel + 255) / 256, 256, 0, stream.stream()>>>(
      numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float *a, const float *b,
                           float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] + b[idx];
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cuda(const at::Tensor &a,
                    const at::Tensor &b,
                    at::Tensor &out) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(b.sizes().equals(out.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(a.device().type() == at::kCUDA);
  TORCH_CHECK(b.device().type() == at::kCUDA);
  TORCH_CHECK(out.device().type() == at::kCUDA);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = out.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  auto stream = at::cuda::getCurrentCUDAStream();

  add_kernel<<<(numel + 255) / 256, 256, 0, stream.stream()>>>(
      numel, a_ptr, b_ptr, result_ptr);
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(cudakit, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
}

} // namespace cudakit
