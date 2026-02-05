#include <Python.h>

#include <torch/extension.h>
#include <torch/library.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace cudakit {

at::Tensor mymuladd_cpu(
    const at::Tensor& a,
    const at::Tensor& b,
    double c) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(a.device().type() == at::kCPU);
  TORCH_CHECK(b.device().type() == at::kCPU);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

at::Tensor mymul_cpu(
    const at::Tensor& a,
    const at::Tensor& b) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(a.device().type() == at::kCPU);
  TORCH_CHECK(b.device().type() == at::kCPU);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out) {
  TORCH_CHECK(a.sizes().equals(b.sizes()));
  TORCH_CHECK(b.sizes().equals(out.sizes()));
  TORCH_CHECK(a.scalar_type() == at::kFloat);
  TORCH_CHECK(b.scalar_type() == at::kFloat);
  TORCH_CHECK(out.scalar_type() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(a.device().type() == at::kCPU);
  TORCH_CHECK(b.device().type() == at::kCPU);
  TORCH_CHECK(out.device().type() == at::kCPU);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = out.mutable_data_ptr<float>();

  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// Defines the operators
TORCH_LIBRARY(cudakit, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(cudakit, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
}

}
