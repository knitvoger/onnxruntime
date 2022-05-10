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

class FastRNN final : public OpKernel {
 public:
  explicit FastRNN(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
