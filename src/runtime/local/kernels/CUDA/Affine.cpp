/*
 * Copyright 2021 The DAPHNE Consortium
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

#include "Affine.h"
#include "HostUtils.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

template<typename T>
static void launch_cublas_gemm(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2, const T* alpha, const T* beta,
                               const T* d_lhs, const T* d_rhs, T* d_res);

template<>
[[maybe_unused]] void launch_cublas_gemm<float>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
        const float* alpha,    const float* beta, const float* d_lhs, const float* d_rhs, float* d_res) {
    CHECK_CUBLAS(cublasSgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
            nc1, beta, d_res, nc2));
}

template<>
[[maybe_unused]] void launch_cublas_gemm<double>(const CUDAContext& ctx, size_t nr1, size_t nc1, size_t nc2,
        const double* alpha, const double* beta, const double* d_lhs, const double* d_rhs, double* d_res) {
    CHECK_CUBLAS(cublasDgemm(ctx.getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nc2, nr1, nc1, alpha, d_rhs, nc2, d_lhs,
                             nc1, beta, d_res, nc2));
}

namespace CUDA::NN::Affine {
    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *weights, const DTArg *bias, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        using VT = typename DTRes::VT;
        const size_t nr1 = data->getNumRows();
        const size_t nc1 = data->getNumCols();
        const size_t nc2 = weights->getNumCols();
        const VT blend_alpha = 1;
        VT blend_beta = 0;
        const VT* d_input = data->getValues(&alloc_desc);
        const VT* d_weights = weights->getValues(&alloc_desc);

        if (nc1 != weights->getNumRows()) {
            throw std::runtime_error(fmt::format("NN::Affine: #cols of lhs and #rows of rhs must be the same ({} != {})",
                    nc1, weights->getNumRows()));
        }

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false, &alloc_desc);
        VT* d_res = res->getValues(&alloc_desc);

        // reverse order to accommodate cublas' col major format (-> res = rhs * lhs)
        launch_cublas_gemm<VT>(*ctx, nr1, nc1, nc2, &blend_alpha, &blend_beta, d_input, d_weights, d_res);

        if(bias) {
            assert((bias->getNumRows() == 1) && "bias dimensions not matching up with weights matrix (W[MxN] -> b[1xN]");
            const VT* d_bias = bias->getValues(&alloc_desc);
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
                    1, bias->getNumCols(), 1, 1));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
                    nr1, nc2, 1, 1));
            blend_beta = 1;
            CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
                    ctx->dst_tensor_desc, d_res));
        }
    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}
