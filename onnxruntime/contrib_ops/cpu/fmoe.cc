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

Status FMoE::Compute(OpKernelContext* context) const {
    const auto* X = context->Input<Tensor>(0);
    const auto* W = context->Input<Tensor>(1);
    const auto* B = context->Input<Tensor>(2);
    const auto* input_num_expert = context->Input<Tensor>(3);
    const auto* input_top_k = context->Input<Tensor>(4);
    const auto* input_gate_index = context->Input<Tensor>(5);
    const auto* input_gate_score = context->Input<Tensor>(6);

    // Dimensions
    int64_t sequence = X->Shape()[2];
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

    //DumpCPU("onnx.input.txt", Xdata, 98*384, 384);
    // Output
    std::vector<int64_t> Y_dims({X->Shape()[0], out_chs, sequence * 2});
    Tensor* Y = context->Output(0, Y_dims);
    float* Ydata = Y->template MutableData<float>();
    //memset(Ydata, 0, sizeof(float) * sequence * out_chs);

    ONNX_UNUSED_PARAMETER(Xdata);
    ONNX_UNUSED_PARAMETER(Wdata);
    ONNX_UNUSED_PARAMETER(Bdata);
    ONNX_UNUSED_PARAMETER(gate_index);
    ONNX_UNUSED_PARAMETER(gate_score);
    ONNX_UNUSED_PARAMETER(sequence);
    ONNX_UNUSED_PARAMETER(in_chs);
    ONNX_UNUSED_PARAMETER(out_chs);
    ONNX_UNUSED_PARAMETER(Ydata);
    ONNX_UNUSED_PARAMETER(num_expert);

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    /*auto* output_k_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * sequence * top_k * out_chs);
    BufferUniquePtr output_k_buffer = BufferUniquePtr(output_k_data, BufferDeleter(alloc));
    float* output_k_buffer_data = static_cast<float*>(output_k_buffer.get());*/

    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
    MLAS_ACTIVATION activation;
    activation.ActivationKind = MlasIdentityActivation;
    TensorShape input_shape = X->Shape().Slice(2);
    TensorShape output_shape = Y->Shape().Slice(2);
    std::vector<int64_t> kernel_shape(1, 1);
    std::vector<int64_t> pads(2, 0);
    std::vector<int64_t> dilations(1, 1);
    std::vector<int64_t> strides(1, 1);


    for (int64_t k = 0; k < top_k; k++)
    {
        float *output_k = Ydata + k * sequence * out_chs;
        const int64_t *gate_index_k = gate_index + k * sequence;
        
        const float *weight = Wdata + gate_index_k[0] * in_chs * out_chs;
        const float *bias = Bdata + gate_index_k[0] * out_chs;
        /*for (int i = 0; i < sequence; i++)
        {
            const float *weight = Wdata + gate_index_k[0] * in_chs * out_chs;
            const float *bias = Bdata + gate_index_k[0] * out_chs;
            math::MatMul<float>(
                sequence,
                out_chs,
                in_chs,
                Xdata,
                weight,
                output_k,
                thread_pool);

            MlasActivation(&activation, output_k, bias, out_chs, sequence,  sequence);
        }*/

        MLAS_CONV_PARAMETERS Parameters;
        size_t WorkingBufferSize;
        MlasConvPrepare(&Parameters,
                        1,
                        1,
                        1,
                        in_chs,
                        input_shape.GetDims().data(),
                        kernel_shape.data(),
                        dilations.data(),
                        pads.data(),
                        strides.data(),
                        output_shape.GetDims().data(),
                        out_chs,
                        &activation,
                        &WorkingBufferSize,
                        thread_pool);

        auto* working_data = WorkingBufferSize > 0 ? alloc->Alloc(SafeInt<size_t>(sizeof(float)) * WorkingBufferSize)
                                                : nullptr;
        BufferUniquePtr working_buffer(working_data, BufferDeleter(alloc));

        MlasConv(&Parameters,
                Xdata,
                weight,
                bias,
                static_cast<float*>(working_buffer.get()),
                output_k,
                thread_pool);
    }



    

    //printf("num_expert %ld, top_k %ld\n", num_expert, top_k);
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
