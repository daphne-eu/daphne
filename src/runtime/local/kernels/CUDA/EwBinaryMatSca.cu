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

#include "EwBinaryMatSca.h"
#include "bin_ops.cuh"

template<class VT, class OP>
__global__ void ewBinMatSca(VT* res, const VT* lhs, const VT rhs, size_t N, OP op) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        res[tid] = op(lhs[tid], rhs);
}

namespace CUDA {
    template<typename VT>
    void EwBinaryMatSca<DenseMatrix<VT>, DenseMatrix<VT>, VT>::apply(BinaryOpCode opCode, DenseMatrix<VT> *&res,
            const DenseMatrix<VT> *lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false, ALLOCATION_TYPE::CUDA_ALLOC);

        auto N = res->getNumItems();

        // ToDo: use templates instead of this if-else madness
        if (opCode == BinaryOpCode::ADD) {
            SumOp<VT> op;
            // auto ctx = dctx->getCUDAContext(0);

            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatSca<VT, SumOp<VT>>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMatSca<<<gridSize, blockSize>>>(res->getValuesCUDA(), lhs->getValuesCUDA(), rhs, N, op);
        }
        else if (opCode == BinaryOpCode::DIV) {
            DivOp<VT> op;
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatSca<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMatSca<<<gridSize, blockSize>>>(res->getValuesCUDA(), lhs->getValuesCUDA(), rhs, N, op);
        }
        else if (opCode == BinaryOpCode::POW) {
            PowOp<VT> op;
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatSca<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMatSca<<<gridSize, blockSize>>>(res->getValuesCUDA(), lhs->getValuesCUDA(), rhs, N, op);
        }
        else if (opCode == BinaryOpCode::SUB) {
            MinusOp<VT> op;
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatSca<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMatSca<<<gridSize, blockSize>>>(res->getValuesCUDA(), lhs->getValuesCUDA(), rhs, N, op);
        }

        else {
            std::cerr << "opCode=" << static_cast<uint32_t>(opCode) << std::endl;
            throw std::runtime_error("unknown operator for EwBinaryMatSca");
        }
    }
    template struct EwBinaryMatSca<DenseMatrix<double>, DenseMatrix<double>, double>;
    template struct EwBinaryMatSca<DenseMatrix<float>, DenseMatrix<float>, float>;
    template struct EwBinaryMatSca<DenseMatrix<int64_t>, DenseMatrix<int64_t>, int64_t>;
}