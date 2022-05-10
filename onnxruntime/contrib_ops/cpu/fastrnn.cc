/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "fastrnn.h"

#include "core/common/safeint.h"
#include "core/util/math_cpuonly.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {

Status FastRNN::Compute(OpKernelContext* context) const {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "");
}

ONNX_OPERATOR_KERNEL_EX(
    FastRNN,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FastRNN);

}
}  // namespace onnxruntime
