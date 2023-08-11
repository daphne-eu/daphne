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

#include "ColBind.h"
#include <runtime/local/datastructures/AllocationDescriptorCUDA.h>

#include <cstdint>

namespace CUDA {

    template<typename T>
    __global__ void cbind(const T *A, const T *B, T *C, int rowsA, int colsA, int rowsB, int colsB) {
        auto maxClen = max(colsA, colsB);
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto ix = tid / maxClen;
        auto iy = tid % maxClen;

        int colsC = colsA + colsB;

        // Copy an element of A into C into the appropriate location
        if(ix < rowsA && iy < colsA) {
            T elemA = A[ix * colsA + iy];
            C[ix * colsC + iy] = elemA;
        }

        // Copy an element of B into C into the appropriate location
        if(ix < rowsB && iy < colsB) {
            T elemB = B[ix * colsB + iy];
            C[ix * colsC + (iy + colsA)] = elemB;
        }
    }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------
    template<typename VTres, typename VTlhs, typename VTrhs>
    void ColBind<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>>::apply(DenseMatrix<VTres> *& res,
            const DenseMatrix<VTlhs> * lhs, const DenseMatrix<VTrhs> * rhs, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs + numColsRhs, false,
                    &alloc_desc);
        }
        auto N = res->getNumItems();
        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;
        CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, cbind<VTres>, 0, 0));
        gridSize = (N + blockSize - 1) / blockSize;
        spdlog::get("runtime::cuda")->debug("ColBind: {} blocks x {} threads = {} total threads for {} items", gridSize,
                blockSize, gridSize*blockSize,N);
        cbind<<<gridSize, blockSize>>>(lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc), res->getValues(&alloc_desc),
                numRowsLhs, numColsLhs, numRowsRhs, numColsRhs);
    }
    template struct ColBind<DenseMatrix<int64_t>, DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
    template struct ColBind<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct ColBind<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
}