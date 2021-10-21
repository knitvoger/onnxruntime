// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
namespace contrib {

class FMoE final : public OpKernel {
 public:
  explicit FMoE(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
  Status ExpertConv(OpKernelContext* context, const float *input, int64_t start_index, int64_t end_index, int64_t in_chs, int64_t out_chs, 
                const float *Wdata, const float *Bdata, int64_t gate_index, float *output, concurrency::ThreadPool* thread_pool) const;
};

}  // namespace contrib
}  // namespace onnxruntime
