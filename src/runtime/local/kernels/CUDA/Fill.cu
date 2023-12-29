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

#include "Fill.h"

#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"

namespace CUDA {

    template<class VT>
    __global__ void fill_kernel(VT *res, VT value, size_t N) {
        for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
            res[i] = value;
    }

    template<typename VT>
    void Fill<DenseMatrix<VT>, VT>::apply(DenseMatrix<VT> *&res, VT arg, size_t numRows, size_t numCols, DCTX(dctx)) {

        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0, &alloc_desc);
        if(arg != 0) {
            fill_kernel<<<ctx->getDeviceProperties()->multiProcessorCount, ctx->default_block_size>>>(
                    res->getValues(&alloc_desc), arg, res->getNumItems());
        }
    }

    template struct Fill<DenseMatrix<float>, float>;
    template struct Fill<DenseMatrix<double>, double>;
    template struct Fill<DenseMatrix<int64_t>, int64_t>;
    template struct Fill<DenseMatrix<uint8_t>, uint8_t>;
    template struct Fill<DenseMatrix<uint64_t>, uint64_t>;
}