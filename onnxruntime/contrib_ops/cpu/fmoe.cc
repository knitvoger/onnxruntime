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

    /*auto* output_k_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * sequence * top_k * out_chs);
    BufferUniquePtr output_k_buffer = BufferUniquePtr(output_k_data, BufferDeleter(alloc));
    float* output_k_buffer_data = static_cast<float*>(output_k_buffer.get());*/

    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
    /*MLAS_ACTIVATION activation;
    activation.ActivationKind = MlasIdentityActivation;
    std::vector<int64_t> kernel_shape(1, 1);
    std::vector<int64_t> pads(2, 0);
    std::vector<int64_t> dilations(1, 1);
    std::vector<int64_t> strides(1, 1);*/

    for (int64_t k = 0; k < top_k; k++)
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
