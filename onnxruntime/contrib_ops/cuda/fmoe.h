// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
namespace contrib {
namespace cuda {

class FMoE final : public onnxruntime::cuda::CudaKernel {
 public:
  using Base = onnxruntime::cuda::CudaKernel;
  explicit FMoE(const OpKernelInfo& info) : onnxruntime::cuda::CudaKernel (info) {
      trans_A_ = false;
      trans_B_ = true;
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  template <typename T>
  Status FMoEImpl(OpKernelContext* context) const;
  template <typename T>
  Status ExpertConv(OpKernelContext* context, const T *input, int64_t start_index, int64_t end_index, int64_t in_chs, int64_t out_chs, 
                const T *Wdata, const T *Bdata, int64_t gate_index, T *output) const;

  ~FMoE() { }

 private:
  int64_t num_expert = 16;
  int64_t top_k  = 2;
  int64_t num_repeat = 1;
  bool trans_A_;
  bool trans_B_;
};

}
}  // namespace contrib
}  // namespace onnxruntime
