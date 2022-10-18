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

#include "BiasAdd.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

namespace CUDA::BiasAdd {
    template<typename DTRes, typename DTArg>
    void Forward<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *bias, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        using VT = typename DTRes::VT;
        const size_t nr1 = data->getNumRows();
        const size_t nc1 = data->getNumCols();
        const VT blend_alpha = 1;
        const VT blend_beta = 1;
        const VT* d_input = data->getValues(&alloc_desc);
        const VT* d_bias = bias->getValues(&alloc_desc);
        res = const_cast<DTArg*>(data);
        VT* d_res = const_cast<VT*>(d_input);

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(), nr1, nc1, 1, 1));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->template getCUDNNDataType<VT>(),
                nr1, nc1, 1, 1));

        CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &blend_alpha, ctx->src_tensor_desc, d_bias, &blend_beta,
                ctx->dst_tensor_desc, d_res));
    }

    template struct Forward<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Forward<DenseMatrix<double>, DenseMatrix<double>>;
}
