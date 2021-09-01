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

#include "fmoe.h"

#include "core/common/safeint.h"
#include "core/util/math_cpuonly.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status FMoE::Compute(OpKernelContext* context) const {
    printf("FMoE cuda!!!!!!!\n");
    return Status::OK();
}

/*ONNX_OPERATOR_KERNEL_EX(
    FMoE,  //name
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    FMoE);*/


ONNX_OPERATOR_TYPED_KERNEL_EX(
    FMoE,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    FMoE);

}
}
}  // namespace onnxruntime
