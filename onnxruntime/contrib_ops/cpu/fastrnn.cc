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

void Gemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                          int64_t M, int64_t N, int64_t K,
                          float alpha,
                          const float* a_data, const float* b_data,
                          float beta,
                          const float* c_data, const TensorShape* c_shape,
                          float* y_data,
                          concurrency::ThreadPool* thread_pool) {
  // if input is empty tensor, return directly as nothing need to be calculated.
  if (M == 0 || N == 0)
    return;

  math::Gemm<float>(trans_a, trans_b,
                M, N, K,
                alpha,
                a_data,
                b_data,
                // ideally we need to set the output buffer contents to 0 if bias is missing,
                // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
                c_data != nullptr ? beta : 0,
                y_data,
                thread_pool);
}

Status FastRNN::Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    // const auto* hiddenStates = context->Input<Tensor>(1);
    const auto* W = context->Input<Tensor>(2);
    const auto* U = context->Input<Tensor>(3);
    // const auto* bias_gate = context->Input<Tensor>(4);
    // const auto* bias_update = context->Input<Tensor>(5);
    // const auto* zeta = context->Input<Tensor>(6);
    // const auto* nu = context->Input<Tensor>(7);


    // Dimensions
    int64_t sequence = X->Shape()[1];
    int64_t in_chs = X->Shape()[2];
    int64_t out_chs = W->Shape()[1];

    // output
    std::vector<int64_t> Y_dims({1, sequence, out_chs});
    Tensor* Y = context->Output(0, Y_dims);

    const float *Xdata = X->template Data<float>();
    const float *Wdata = W->template Data<float>();
    const float *Udata = U->template Data<float>();
    float* Ydata = Y->template MutableData<float>();
    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

    float *buffer  =(float*)malloc(sizeof(float) * 3 * out_chs);
    float *wComp = buffer;
    float *uComp = buffer + out_chs;
    float *preComp = buffer + 2 * out_chs;

    for (int i = 0; i < sequence; i++)
    {
      const float *Xdata_i = Xdata + i * in_chs;
      float *Ydata_i = Ydata + i * out_chs;

      int64_t M = 1;
      int64_t N = out_chs;
      int64_t K = in_chs;
      TensorShape state_shape = {out_chs};

      Gemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0, Xdata_i, Wdata, 1.0,
                  nullptr, &state_shape, wComp, thread_pool);

      Gemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0, Xdata_i, Udata, 1.0,
                  nullptr, &state_shape, uComp, thread_pool);

      MLAS_ACTIVATION act;
      act.ActivationKind = MlasLogisticActivation;
      MlasActivation(
        &act,
        wComp,
        nullptr,
        1,
        N,
        N
      ); 

      act.ActivationKind = MlasTanhActivation;
      MlasActivation(
        &act,
        uComp,
        nullptr,
        1,
        N,
        N
      ); 

      //math::Add<float>(wComp, uComp, Ydata_i);
      ONNX_UNUSED_PARAMETER(Ydata_i);
    }

    free(buffer);

    
    ONNX_UNUSED_PARAMETER(Udata);
    ONNX_UNUSED_PARAMETER(uComp);
    ONNX_UNUSED_PARAMETER(preComp);
    return Status::OK();
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
