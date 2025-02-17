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

namespace CUDA {
template <typename DTRes, typename DTArg>
void BiasAdd<DTRes, DTArg>::apply(DTRes *&res, const DTArg *data, const DTArg *bias, DCTX(dctx)) {
    const size_t deviceID = 0; // TODO: multi device support
    auto *ctx = CUDAContext::get(dctx, deviceID);
    AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

    using VT = typename DTRes::VT;
    const size_t nr1 = data->getNumRows();
    const size_t nc1 = data->getNumCols();

    const VT *d_input = data->getValues(&alloc_desc);
    const VT *d_bias = bias->getValues(&alloc_desc);

    if (!res)
        res = DataObjectFactory::create<DTRes>(nr1, nc1, /*zero=*/false, &alloc_desc);
    VT *d_res = res->getValues(&alloc_desc);

    CHECK_CUDART(cudaMemcpy(d_res, d_input, nr1 * nc1 * sizeof(VT), cudaMemcpyDeviceToDevice));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
                                           nr1, // batch size
                                           nc1, // channels
                                           1, 1));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->getCUDNNDataType<VT>(),
                                           nr1, // batch size
                                           nc1, // channels
                                           1, 1));

    constexpr VT alpha = 1;
    constexpr VT beta = 1;
    // res = data
    // res = alpha*bias + beta*res
    CHECK_CUDNN(cudnnAddTensor(ctx->getCUDNNHandle(), &alpha, ctx->src_tensor_desc, d_bias, &beta, ctx->dst_tensor_desc,
                               d_res));
}

template struct BiasAdd<DenseMatrix<float>, DenseMatrix<float>>;
template struct BiasAdd<DenseMatrix<double>, DenseMatrix<double>>;

} // namespace CUDA
