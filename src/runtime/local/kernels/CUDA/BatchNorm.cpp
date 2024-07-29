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

namespace CUDA::NN::BatchNorm {
    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *gamma, const DTArg *beta,
            const size_t num_channels, const size_t img_h, const size_t img_w, const bool train,
            DTArg *ema_mean, DTArg *ema_var, double mu, double eps, DCTX(dctx))
    {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        using VT = typename DTRes::VT;
        VT blend_alpha = 1.0;
        VT blend_beta = 0.0;

        const size_t N = data->getNumRows();
        const void* d_input = data->getValues(&alloc_desc);
        const void* d_gamma = gamma->getValues(&alloc_desc);
        const void* d_beta = beta->getValues(&alloc_desc);
        VT* d_ema_mean = ema_mean->getValues(&alloc_desc);
        VT* d_ema_var = ema_var->getValues(&alloc_desc);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
                N, num_channels, img_h, img_w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
                N, num_channels, img_h, img_w));

        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(N, num_channels * img_w * img_h, false, &alloc_desc);
        }

        VT* d_res = res->getValues(&alloc_desc);
        CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(ctx->bn_tensor_desc, ctx->src_tensor_desc, ctx->bn_mode));

        if(train) {
            //ToDo: resultSaveMean, resultSaveInvVariance
            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(ctx->getCUDNNHandle(), ctx->bn_mode, &blend_alpha,
                                                               &blend_beta, ctx->src_tensor_desc, d_input, ctx->dst_tensor_desc, d_res, ctx->bn_tensor_desc,
                                                               d_gamma, d_beta, mu, d_ema_mean, d_ema_var, eps, nullptr,
                                                               nullptr));
        }
        else {
            CHECK_CUDNN(cudnnBatchNormalizationForwardInference(ctx->getCUDNNHandle(), ctx->bn_mode, &blend_alpha,
                    &blend_beta, ctx->src_tensor_desc, d_input, ctx->dst_tensor_desc, d_res, ctx->bn_tensor_desc,
                    d_gamma, d_beta, d_ema_mean, d_ema_var, eps));
        }
    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}

