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


void DumpCPU(const char *file, const float *cpu_data, long size, long row)
{
	FILE *pFile = fopen(file, "w");

	if (pFile)
	{
		for (long i = 0; i < size; i++)
		{
			fprintf(pFile, "%.5lf ", (cpu_data[i]));
			if ((i + 1) % row == 0)
			{
				fprintf(pFile, "\n");
			}
		}

	}

	fclose(pFile);
}

static void GemmBroadcastBias(int64_t M, int64_t N, float beta,
                              const float* c_data, const TensorShape* c_shape,
                              float* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<float>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<float>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<float>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<float>(c_data, M, N);
    }
  }
}

void ComputeGemm(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
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

  // Broadcast the bias as needed if bias is given
  GemmBroadcastBias(M, N, beta, c_data, c_shape, y_data);

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

Status FMoE:: ExpertConv(OpKernelContext* context, const float *input, int64_t start_index, int64_t end_index, int64_t in_chs, int64_t out_chs, 
                const float *Wdata, const float *Bdata, int64_t gate_index, float *output, concurrency::ThreadPool* thread_pool) const
{
    const float *weight = Wdata + gate_index * in_chs * out_chs;
    const float *bias = Bdata + gate_index * out_chs;

    MLAS_ACTIVATION activation;
    activation.ActivationKind = MlasIdentityActivation;

    /*math::MatMul<float>(
        end_index - start_index,
        out_chs,
        in_chs,
        input,
        weight,
        output,
        thread_pool);*/

    int64_t M = end_index - start_index;
    int64_t N = out_chs;
    int64_t K = in_chs;
    TensorShape bias_shape = {N};
    //MlasGemm(CblasNoTrans, CblasTrans, M, N, K, 1.f, input, K, weight, K, 0.f, output, N, thread_pool);
    //MlasActivation(&activation, output, bias, out_chs, end_index - start_index,  end_index - start_index);

    ComputeGemm(CblasNoTrans, CblasTrans, M, N, K, 1.0, input, weight, 1.0,
                bias, &bias_shape, output, thread_pool);

    ONNX_UNUSED_PARAMETER(bias);
    ONNX_UNUSED_PARAMETER(activation);
    return Status::OK();
}

Status FMoE::Compute(OpKernelContext* context) const {
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
    const int64_t num_expert = *(input_num_expert->template Data<int64_t>());
    const int64_t top_k = *(input_top_k->template Data<int64_t>());
    const int64_t *gate_index = input_gate_index->template Data<int64_t>();
    const float *gate_score= input_gate_score->template Data<float>();
    const int64_t num_repeat= *(input_num_repeat->template Data<int64_t>());

    //DumpCPU("onnx.input.txt", Xdata, 98*384, 384);
    // Output
    std::vector<int64_t> Y_dims({num_repeat == 1 ? sequence * top_k : sequence, out_chs});
    Tensor* Y = context->Output(0, Y_dims);
    float* Ydata = Y->template MutableData<float>();
    //memset(Ydata, 0, sizeof(float) * sequence * out_chs);

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

    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

    // phones one by one
    /*for (int64_t k = 0; k < top_k; k++)
    {
        int64_t seq_k = num_repeat == 1 ? sequence : sequence / top_k;
        const float *input_x = num_repeat == 1 ? Xdata : (Xdata + k * seq_k * in_chs);
        float *output_k = Ydata + k * seq_k * out_chs;
        const int64_t *gate_index_k = gate_index + k * seq_k;

        int64_t i = 0;
        while(i < sequence)
        {
            int64_t end_index = i + 1;
            while(end_index < sequence && gate_index_k[i] == gate_index_k[end_index]){
                end_index++;
            }
            
            // conv for input[:, i:end_index]
            this->ExpertConv(context, input_x + i * in_chs, i, end_index, in_chs, out_chs, Wdata, Bdata, gate_index_k[i], output_k + i * out_chs, thread_pool);
            i = end_index;
        }
  
        if (num_repeat == 0)
            break; 
    }*/

    // phones with same expert
    int64_t total_processed = 0;
    int64_t threshold = 12;
    std::vector<Expert_Run_Parameter> expert_run_params;
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
            expert_run_params.clear();
            int64_t gate_to_process = *it;
            for (int64_t i =0; i < sequence; i++)
            {
                if (gate_index_k[i] != gate_to_process)
                    continue;

                int64_t end_index = i + 1;  
                while(end_index < sequence && gate_index_k[i] == gate_index_k[end_index]){
                    end_index++;
                }
                
                // conv for input[:, i:end_index]
                if (end_index - i >= threshold){
                    this->ExpertConv(context, input_x + i * in_chs, i, end_index, in_chs, out_chs, Wdata, Bdata, gate_index_k[i], output_k + i * out_chs, thread_pool);
                }
                else{
                    int64_t M = end_index - i;
                    int lda = static_cast<int>((TransA == CblasNoTrans) ? K : M);
                    int ldb = static_cast<int>((TransB == CblasNoTrans) ? N : K);

                    Expert_Run_Parameter param;
                    param.mlas_params.A = input_x + i * in_chs;
                    param.mlas_params.alpha = 1.0;
                    param.mlas_params.B = Wdata + gate_index_k[i] * in_chs * out_chs;
                    param.mlas_params.beta = 1.0;
                    param.mlas_params.BIsPacked = false;
                    param.mlas_params.C = output_k + i * out_chs;
                    param.mlas_params.lda = lda;
                    param.mlas_params.ldb = ldb;
                    param.mlas_params.ldc = N;
                    param.M = M;
                    param.N = N;
                    param.K = K;
                    param.bias = Bdata + gate_index_k[i] * out_chs;

                    TensorShape bias_shape = {N};
                    GemmBroadcastBias(M, N, 1.0, param.bias , &bias_shape, param.mlas_params.C);
                    expert_run_params.push_back(param);
                }   
                total_processed += end_index - i;
                i = end_index;
            }

            // run experts with small input (< threshold)
            /*const int64_t ThreadPerGemm = 2;
            MlasTrySimpleParallel(thread_pool, 
                ThreadPerGemm * static_cast<ptrdiff_t>(expert_run_params.size()), 
                [=](ptrdiff_t tid)
            {
                ptrdiff_t GemmIdx = tid / ThreadPerGemm;
                ptrdiff_t ThreadIdx = tid % ThreadPerGemm;
                MlasSgemmThreaded(1, 2, TransA, TransB, 
                    expert_run_params[GemmIdx].M, expert_run_params[GemmIdx].N, expert_run_params[GemmIdx].K, 
                    &(expert_run_params[GemmIdx]), ThreadIdx);
            });*/
            if (expert_run_params.size() > 0)
                MlasGemmBatch(TransA, TransB, expert_run_params, thread_pool);
        }
  
        if (num_repeat == 0)
            break; 
    }

    if (total_processed != Y_dims[0]){
        printf("Unexpected!!!\n");
        exit(0);
    }

    /*static int c = 0;
    char file[1024];
    sprintf(file, "fmoe_dump/input_%d.txt", c);
    //printf("num_expert %ld, top_k %ld\n", num_expert, top_k);
    DumpCPU(file, Xdata, in_chs * sequence, in_chs);
    sprintf(file, "fmoe_dump/output_%d.txt", c);
    DumpCPU(file, Ydata, Y_dims[0] * Y_dims[1], Y_dims[1]);
    c++;*/

    return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    FMoE,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    FMoE);

}
}  // namespace onnxruntime
