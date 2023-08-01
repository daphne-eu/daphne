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

#include "AggCol.h"
#include "HostUtils.h"

#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

#include "bin_ops.cuh"

namespace CUDA {

    template<class VT, class OP>
    __global__ void agg_col(VT* res, const VT* arg, size_t N, size_t cols, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= cols) {
            return;
        }

        auto i = tid;
        auto grid_size = cols;
        VT val = OP::init();

        while (i < N) {
            val = op.exec(val, arg[i]);
            i += grid_size;
        }
        res[tid] = val;
    }

    template<typename VT>
    void AggCol<DenseMatrix<VT>, DenseMatrix<VT>>::apply(AggOpCode opCode, DenseMatrix<VT> *&res,
            const DenseMatrix<VT> *arg, DCTX(dctx)) {
        const size_t numCols = arg->getNumCols();
        
        const size_t deviceID = 0; //ToDo: multi device support
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(1, numCols, false,  &alloc_desc);

        auto N = arg->getNumItems();

        if (opCode == AggOpCode::SUM) {
            SumOp<VT> op;

            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agg_col<VT, SumOp<VT>>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            agg_col<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), arg->getValues(&alloc_desc), N, numCols, op);
        }
        else if (opCode == AggOpCode::MAX) {
            MaxOp<VT> op;

            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agg_col<VT, MaxOp<VT>>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            agg_col<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), arg->getValues(&alloc_desc), N, numCols, op);
        }
        else if (opCode == AggOpCode::MIN) {
            MinOp<VT> op;

            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agg_col<VT, MinOp<VT>>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            agg_col<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), arg->getValues(&alloc_desc), N, numCols, op);
        }
        else {
            throw std::runtime_error(fmt::format("Unknown opCode {} for aggCol", static_cast<uint32_t>(opCode)));
        }
    }
    template struct AggCol<DenseMatrix<double>, DenseMatrix<double>>;
    template struct AggCol<DenseMatrix<float>, DenseMatrix<float>>;
    template struct AggCol<DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
}

