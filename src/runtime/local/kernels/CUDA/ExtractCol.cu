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

#include "ExtractCol.h"
#include <cstdint>

namespace CUDA {
    template<class DTRes, class DTArg, class DTSel>
    __global__ void extract_col(DTRes *res, const DTArg *arg, const DTSel *sel, const size_t sel_rows, const size_t arg_cols, const size_t cols) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < cols) {
            auto idx = sel[tid%sel_rows];
            auto row = tid / sel_rows;
            res[tid] = arg[row * arg_cols + idx];
        }
    }

    // ----------------------------------------------------------------------------
    // DenseMatrix <- DenseMatrix, DenseMatrix
    // ----------------------------------------------------------------------------
    template<class DTRes, class DTArg, class DTSel>
    void ExtractCol<DenseMatrix<DTRes>, DenseMatrix<DTArg>, DenseMatrix<DTSel>>::apply(DenseMatrix<DTRes>*& res,
            const DenseMatrix<DTArg>* arg, const DenseMatrix<DTSel>* sel, DCTX(ctx)) {

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<DTRes>>(arg->getNumRows(), sel->getNumRows(), false,
                    ALLOCATION_TYPE::CUDA_ALLOC);
        }
        auto N = res->getNumItems();
        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;
        CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, extract_col<DTRes, DTArg, DTSel>, 0, 0));
        gridSize = (N + blockSize - 1) / blockSize;

#ifndef NDEBUG
        std::cout << " ExtractCol: " << gridSize << " blocks x " << blockSize << " threads = " << gridSize*blockSize
                << " total threads for " << N << " items" << std::endl;
#endif

        extract_col<<<gridSize, blockSize>>>(res->getValuesCUDA(), arg->getValuesCUDA(), sel->getValuesCUDA(),
                sel->getNumRows(), arg->getNumCols(), N);


//        std::vector<DTRes> res_host(res->getNumItems());
//        CHECK_CUDART(cudaMemcpy(res_host.data(), res->getValuesCUDA(), res->bufferSize(), cudaMemcpyDeviceToHost));
//        for(auto j=0; j < res->getNumItems(); j++)
//            std::cout << res_host[j] << " ";
//        std::cout << std::endl;
    }
    template struct ExtractCol<DenseMatrix<int64_t>, DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
    template struct ExtractCol<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<int64_t>>;
    template struct ExtractCol<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<int64_t>>;
}