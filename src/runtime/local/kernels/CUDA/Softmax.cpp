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

#include "Softmax.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

namespace CUDA::Softmax {

    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        using VT = typename DTRes::VT;
        int n = data->getNumRows();
        int d = data->getNumCols();
        const VT blend_alpha = 1;
        const VT blend_beta = 0;
        const VT* d_input = data->getValues(&alloc_desc);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), n, d, 1, 1));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), n, d, 1, 1));

        if (res == nullptr) {
            res = DataObjectFactory::create<DTRes>(n,d, false, &alloc_desc);
        }
        VT* d_res = res->getValues(&alloc_desc);

        CHECK_CUDNN(cudnnSoftmaxForward(ctx->getCUDNNHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                &blend_alpha, ctx->src_tensor_desc, d_input, &blend_beta, ctx->dst_tensor_desc, d_res));
    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}

