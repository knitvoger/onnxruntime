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

#define MAX_SEQ_LENGTH 10000*2

class FMoE final : public onnxruntime::cuda::CudaKernel {
 public:
  using Base = onnxruntime::cuda::CudaKernel;
  explicit FMoE(const OpKernelInfo& info) : onnxruntime::cuda::CudaKernel (info) {
      gate_index = (int64_t *)malloc(sizeof(int64_t) * MAX_SEQ_LENGTH);
      gate_score = (float *)malloc(sizeof(float) * MAX_SEQ_LENGTH);
      trans_A_ = false;
      trans_B_ = true;
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ExpertConv(OpKernelContext* context, const float *input, int64_t start_index, int64_t end_index, int64_t in_chs, int64_t out_chs, 
                const float *Wdata, const float *Bdata, int64_t gate_index, float *output) const;

  ~FMoE() { free(gate_index); free(gate_score); }

 private:
  int64_t num_expert = 16;
  int64_t top_k  = 2;
  int64_t num_repeat = 1;
  int64_t *gate_index = nullptr;
  float *gate_score= nullptr;
  bool trans_A_;
  bool trans_B_;
};

}
}  // namespace contrib
}  // namespace onnxruntime
