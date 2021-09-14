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
#include "core/providers/cuda/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include <chrono>

#define ONNX_UNUSED_PARAMETER(x) (void)(x)
namespace onnxruntime {
namespace contrib {
namespace cuda {

typedef typename onnxruntime::cuda::ToCudaType<float>::MappedType CudaT;
static float gemm_total_cost = 0;

Status FMoE:: ExpertConv(OpKernelContext* context, const float *input, int64_t start_index, int64_t end_index, int64_t in_chs, int64_t out_chs, 
                const float *Wdata, const float *Bdata, int64_t gate_index, float *output) const
{
    //auto start_time = std::chrono::system_clock::now();
    const float *weight = Wdata + gate_index * in_chs * out_chs;
    const float *bias = Bdata + gate_index * out_chs;

    int64_t M = end_index - start_index;
    int64_t N = out_chs;
    int64_t K = in_chs;
    TensorShape bias_shape = {N};
    

    auto one = onnxruntime::cuda::ToCudaType<float>::FromFloat(1.0f);
    auto zero = onnxruntime::cuda::ToCudaType<float>::FromFloat(0.0f);
    auto& device_prop = GetDeviceProp();

    // broadcast bias
    
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, 1,
        &one,
        bias, N,
        GetConstOnes<CudaT>(M), 1,
        &zero,
        output, N, device_prop));

    CudaT alpha = ToCudaType<float>::FromFloat(1.0);
    CudaT beta = ToCudaType<float>::FromFloat(1.0);
    
    
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      CublasHandle(),
      trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      N, M, K,
      &alpha,
      reinterpret_cast<const CudaT*>(weight),
      (trans_B_ ? K : N),
      reinterpret_cast<const CudaT*>(input),
      (trans_A_ ? M : K),
      // ideally we need to set the output buffer contents to 0 if bias is missing,
      // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
      bias != nullptr ? &beta : &zero,
      output, N, device_prop));
    //auto end_time = std::chrono::system_clock::now();
    //auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    //printf("cublasGemmHelper cost %.3f ms\n", infer_duration.count() / 1000.0);
    


    ONNX_UNUSED_PARAMETER(context);
    ONNX_UNUSED_PARAMETER(input);
    ONNX_UNUSED_PARAMETER(output);
    ONNX_UNUSED_PARAMETER(weight);
    ONNX_UNUSED_PARAMETER(bias);
    ONNX_UNUSED_PARAMETER(M);
    ONNX_UNUSED_PARAMETER(N);
    ONNX_UNUSED_PARAMETER(K);
    ONNX_UNUSED_PARAMETER(bias_shape);
    ONNX_UNUSED_PARAMETER(one);
    ONNX_UNUSED_PARAMETER(zero);
    ONNX_UNUSED_PARAMETER(device_prop);
    ONNX_UNUSED_PARAMETER(alpha);
    ONNX_UNUSED_PARAMETER(beta);
    //gemm_total_cost += infer_duration.count() / 1000.0;
    return Status::OK();
}

Status FMoE::ComputeInternal(OpKernelContext* context) const {
    gemm_total_cost = 0;
    //int64_t gemm_count = 0;
    //auto start_time = std::chrono::system_clock::now();
    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);
    const auto* B = context->Input<Tensor>(2);
    const auto* input_num_expert = context->Input<Tensor>(3);
    const auto* input_top_k = context->Input<Tensor>(4);
    const auto* input_gate_index = context->Input<Tensor>(5);
    const auto* input_gate_score = context->Input<Tensor>(6);
    const auto* input_num_repeat = context->Input<Tensor>(7);

    // Dimensions
    int64_t sequence = X->Shape()[0];
    int64_t in_chs = X->Shape()[1];
    int64_t out_chs = W->Shape()[1];

    // Inputs
    const float *Xdata = X->template Data<float>();
    const float *Wdata = W->template Data<float>();
    const float *Bdata = B->template Data<float>();
    // const int64_t num_expert = *(input_num_expert->template Data<int64_t>());
    // const int64_t top_k = *(input_top_k->template Data<int64_t>());
    // const int64_t num_repeat = *(input_num_repeat->template Data<int64_t>());
    cudaMemcpyAsync(const_cast<int64_t*>(&num_expert), input_num_expert->template Data<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, nullptr);
    cudaMemcpyAsync(const_cast<int64_t*>(&top_k), input_top_k->template Data<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, nullptr);
    cudaMemcpyAsync(const_cast<int64_t*>(&num_repeat), input_num_repeat->template Data<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost, nullptr);
    

    cudaMemcpyAsync(const_cast<int64_t*>(gate_index), input_gate_index->template Data<int64_t>(), sizeof(int64_t) * sequence * top_k, cudaMemcpyDeviceToHost, nullptr);
    cudaMemcpyAsync(const_cast<float*>(gate_score), input_gate_score->template Data<float>(), sizeof(float) * sequence * top_k, cudaMemcpyDeviceToHost, nullptr);

    //printf("sequence %ld, num_expert %ld, top_k %ld, num_repeat %ld\n", sequence, num_expert, top_k, num_repeat);
    std::vector<int64_t> Y_dims({num_repeat == 1 ? sequence * top_k : sequence, out_chs});
    Tensor* Y = context->Output(0, Y_dims);
    float* Ydata = Y->template MutableData<float>();

    int64_t total_processed = 0;
    int64_t threshold = 12;
    //std::vector<Expert_Run_Parameter> expert_run_params;
    int64_t N = out_chs;
    int64_t K = in_chs;
    CBLAS_TRANSPOSE TransA = CblasNoTrans;
    CBLAS_TRANSPOSE TransB = CblasTrans;
    for (int64_t k = 0; k < top_k; k++)
    {
        int64_t seq_k = num_repeat == 1 ? sequence : sequence / top_k;
        const float *input_x = num_repeat == 1 ? Xdata : (Xdata + k * seq_k * in_chs);
        float *output_k = Ydata + k * seq_k * out_chs;
        const int64_t *gate_index_k = gate_index + k * seq_k;

        std::set<int64_t> unique_gates;
        for (int i = 0; i < sequence; i++)
            unique_gates.insert(gate_index_k[i]);

        for (auto it = unique_gates.begin(); it != unique_gates.end(); ++it)
        {
            //expert_run_params.clear();
            int64_t gate_to_process = *it;
            //printf("gate_to_process %ld\n", gate_to_process);
            for (int64_t i =0; i < sequence; i++)
            {
                if (gate_index_k[i] != gate_to_process)
                    continue;

                int64_t end_index = i + 1;  
                while(end_index < sequence && gate_index_k[i] == gate_index_k[end_index]){
                    end_index++;
                }
                
                // conv for input[:, i:end_index]
                this->ExpertConv(context, input_x + i * in_chs, i, end_index, in_chs, out_chs, Wdata, Bdata, gate_index_k[i], output_k + i * out_chs);
                //gemm_count++;
                total_processed += end_index - i;
                i = end_index;
            }
        }
  
        ONNX_UNUSED_PARAMETER(input_x);
        ONNX_UNUSED_PARAMETER(output_k);
        if (num_repeat == 0)
            break; 
    }

    /*printf("gemm_total_cost %.3f\n", gemm_total_cost);
    auto end_time = std::chrono::system_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("fmoe cost  %.3f\n", infer_duration.count() / 1000.0);
    printf("seq %ld, gemm_count %ld\n", sequence, gemm_count);*/

    ONNX_UNUSED_PARAMETER(total_processed);
    ONNX_UNUSED_PARAMETER(threshold);
    ONNX_UNUSED_PARAMETER(N);
    ONNX_UNUSED_PARAMETER(K);
    ONNX_UNUSED_PARAMETER(TransA);
    ONNX_UNUSED_PARAMETER(TransB);
    ONNX_UNUSED_PARAMETER(Xdata);
    ONNX_UNUSED_PARAMETER(Wdata);
    ONNX_UNUSED_PARAMETER(Bdata);
    ONNX_UNUSED_PARAMETER(gate_index);
    ONNX_UNUSED_PARAMETER(num_repeat);
    ONNX_UNUSED_PARAMETER(sequence);
    ONNX_UNUSED_PARAMETER(in_chs);
    ONNX_UNUSED_PARAMETER(out_chs);
    ONNX_UNUSED_PARAMETER(Ydata);
    ONNX_UNUSED_PARAMETER(gate_score);
    ONNX_UNUSED_PARAMETER(num_expert);
    ONNX_UNUSED_PARAMETER(input_num_expert);
    ONNX_UNUSED_PARAMETER(input_top_k);
    ONNX_UNUSED_PARAMETER(input_num_repeat);
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
