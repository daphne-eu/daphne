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

#include "BatchNorm.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>
#include <iostream>

namespace CUDA::BatchNorm {
    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *gamma, const DTArg *beta,
                                      const DTArg *ema_mean, const DTArg *ema_var, const typename DTArg::VT eps, DCTX(dctx))
    {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        using VT = typename DTRes::VT;
        const size_t nr1 = data->getNumRows();
        const size_t nc1 = data->getNumCols();
        VT blend_alpha = 1.0;
        VT blend_beta = 0.0;
        const VT* d_input = data->getValues(&alloc_desc);
        const VT* d_gamma = gamma->getValues(&alloc_desc);
        const VT* d_beta = beta->getValues(&alloc_desc);
        const VT* d_ema_mean = ema_mean->getValues(&alloc_desc);
        const VT* d_ema_var = ema_var->getValues(&alloc_desc);
        size_t num_channels = gamma->getNumRows();

        size_t HW = nc1 / num_channels;
        auto H = static_cast<size_t>(std::sqrt(HW));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), nr1, num_channels, H, H));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), nr1, num_channels, H, H));

        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(nr1, nc1, false, &alloc_desc);
        }
        VT* d_res = res->getValues(&alloc_desc);
        CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(ctx->bn_tensor_desc, ctx->src_tensor_desc, ctx->bn_mode));
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(ctx->getCUDNNHandle(), ctx->bn_mode, &blend_alpha,
                &blend_beta, ctx->src_tensor_desc, d_input, ctx->dst_tensor_desc, d_res, ctx->bn_tensor_desc,
                d_gamma, d_beta, d_ema_mean, d_ema_var, eps));
    }

    template<typename DTRes, typename DTArg>
    void Backward<DTRes, DTArg>::apply(DTRes *&dX, DTRes *&dGamma, DTRes *&dBeta,
                                       const DTArg *mean, const DTArg *invVar, 
                                       const DTArg *in, const DTArg *dout, 
                                       const DTArg *gamma, const typename DTArg::VT eps, DCTX(dctx))
    {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        using VT = typename DTRes::VT;
        const size_t N = in->getNumRows();
        const size_t CHW = in->getNumCols();
        const size_t C = gamma->getNumRows();
        const size_t HW = CHW / C;
        auto H = static_cast<size_t>(std::sqrt(HW));

        VT alphaDataDiff = 1.0;
        VT betaDataDiff = 0.0;
        VT alphaParamDiff = 1.0;
        VT betaParamDiff = 0.0;

        const VT* d_mean = mean->getValues(&alloc_desc);
        const VT* d_invVar = invVar->getValues(&alloc_desc);
        const VT* d_in = in->getValues(&alloc_desc);
        const VT* d_gamma = gamma->getValues(&alloc_desc);
        const VT* d_dout = dout->getValues(&alloc_desc);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), N, C, H, H));    
//        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dy_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), N, C, H, H));
        
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), N, C, H, H));
        CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(ctx->bn_scale_bias_tensor_desc, ctx->src_tensor_desc, ctx->bn_mode));

        if (dX == nullptr)
            dX = DataObjectFactory::create<DenseMatrix<VT>>(N, CHW, false, &alloc_desc);
        if (dGamma == nullptr)
            dGamma = DataObjectFactory::create<DenseMatrix<VT>>(C, 1, false, &alloc_desc);
        if (dBeta == nullptr)
            dBeta = DataObjectFactory::create<DenseMatrix<VT>>(C, 1, false, &alloc_desc);

        VT* d_dX = dX->getValues(&alloc_desc);
        VT* d_dGamma = dGamma->getValues(&alloc_desc);
        VT* d_dBeta = dBeta->getValues(&alloc_desc);

        CHECK_CUDNN(cudnnBatchNormalizationBackward(ctx->getCUDNNHandle(), 
                                                    ctx->bn_mode,
                                                    &alphaDataDiff, &betaDataDiff, &alphaParamDiff, &betaParamDiff,
                                                    ctx->src_tensor_desc, d_in,
                                                    ctx->dst_tensor_desc, d_dout,
//                                                    ctx->dy_tensor_desc, d_dout,
                                                    ctx->dst_tensor_desc, d_dX,
                                                    ctx->bn_scale_bias_tensor_desc, d_gamma, d_dGamma, d_dBeta,
                                                    eps,
                                                    d_mean, d_invVar));

    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;

    template struct Backward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Backward<DenseMatrix<double>, DenseMatrix<double>>;
}

