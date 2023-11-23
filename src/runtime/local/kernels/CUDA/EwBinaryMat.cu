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

#include "EwBinaryMat.h"
#include "HostUtils.h"
#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"
#include "runtime/local/kernels/CUDA/bin_ops.cuh"

namespace CUDA {
    template<class VT, class OP>
    __global__ void ewBinMat(VT *res, const VT *lhs, const VT *rhs, size_t N, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto ltid = tid;
        if(ltid < N)
            res[ltid] = op(lhs[ltid], rhs[ltid]);
//        if(threadIdx.x < 1)
//            printf("bid=%d ltid=%d lhs=%4.3f rhs=%4.3f res=%4.3f\n", blockIdx.x, ltid, lhs[ltid], rhs[ltid], res[ltid]);
//	}
    }

// Todo: templatize this
    template<class VT, class OP>
    __global__ void ewBinMatRVec(VT *res, const VT *lhs, const VT *rhs, size_t dim, size_t N, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto ltid = tid;
//	while(ltid < N) {
        if(ltid < N) {
//        if(ltid == 9)
//            printf("C ltid=%d dim=%d ltid mod dim=%d\n", ltid, dim, ltid %dim);
            res[ltid] = op(lhs[ltid], rhs[ltid % dim]);
//		if(ltid == 9) {
//			printf("R ltid=%d ltidim=%d\n", ltid, ltid % dim);
//			printf("lhs[ltid]=%f\n",lhs[ltid]);
//			printf("rhs[ltid %% dim]=%f\n", rhs[ltid % dim]);
//			printf("res[ltid]=%f\n", res[ltid]);
//		}
//		ltid += gridDim.x;
        }
    }

    template<class VT, class OP>
    __global__ void ewBinMatCVec(VT *res, const VT *lhs, const VT *rhs, size_t dim, size_t N, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto ltid = tid;
//	while(ltid < N) {
        if(ltid < N) {
//		if(ltid == 9)
//			printf("C ltid=%d ltidim=%d\n", ltid, ltid/dim);
            res[ltid] = op(lhs[ltid], rhs[ltid / dim]);
//		ltid += gridDim.x;
        }
    }

    template<class VT, class OP>
    __global__ void ewBinMatSparseDense(VT *res, const VT *lhs_val, const size_t* lhs_cidxs, const size_t * lhs_rptrs, const VT *rhs_val, size_t N, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;

        auto row_start = lhs_rptrs[blockIdx.x];
        auto row_end = lhs_rptrs[blockIdx.x + 1];
        auto idx = row_start + threadIdx.x;

//        if(tid < N) {
//
//        }

        if(idx < row_end) {
            auto val = lhs_val[idx];
            auto col = lhs_cidxs[idx];
            auto r_val = rhs_val[col];
            res[idx] = op(val, r_val);
//        if(threadIdx.x < 1)
//            printf("bid=%d ltid=%d lhs=%4.3f rhs=%4.3f res=%4.3f\n", blockIdx.x, ltid, lhs[ltid], rhs[ltid], res[ltid]);
	    }
    }

    template<class VT, class OP>
    __global__ void ewBinMatSparseSparse(VT *res, const VT *lhs_val, const size_t* lhs_cidxs, const size_t * lhs_rptrs, const VT *rhs_val, size_t N, OP op) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;

        auto row_start = lhs_rptrs[blockIdx.x];
        auto row_end = lhs_rptrs[blockIdx.x + 1];
        auto idx = row_start + threadIdx.x;

        if(tid < N)
            printf("tid=%d N=%llu\n", tid, N);

        if(idx < row_end) {
            auto val = lhs_val[idx];
            auto col = lhs_cidxs[idx];
//            auto r_val = rhs_val[col];
//            res[idx] = op(val, r_val);
//            if(threadIdx.x < 1)
            printf("bid=%d tid=%d idx=%d lhs_val=%4.3f col_idx=%llu row_start=%llu row_end=%llu row_nnz=%llu N=%llu nnz=%llu\n", blockIdx.x, tid, idx, val, col, row_start, row_end, (row_end - row_start), N, lhs_rptrs[N]);
        }
    }

    template<class VT, class OP>
    bool launch_ewbinmat(const size_t& numRowsLhs, const size_t& numColsLhs, const size_t& numRowsRhs,
            const size_t& numColsRhs, size_t& gridSize, int& minGridSize, int& blockSize, const size_t& N, VT* res, const VT* lhs,
            const VT* rhs) {
        OP op;

        if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMat<<<gridSize, blockSize>>>(res, lhs, rhs, N, op);
        }
        else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatRVec<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;

            ewBinMatRVec<<<gridSize, blockSize>>>(res, lhs, rhs, numColsRhs, N, op);
        }
        else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatCVec<VT, decltype(op)>, 0, 0));
            gridSize = (N + blockSize - 1) / blockSize;
            ewBinMatCVec<<<gridSize, blockSize>>>(res, lhs, rhs, numRowsRhs, N, op);
        }
        else
            return true;
        return false;
    }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------
    template<typename VTres, typename VTlhs, typename VTrhs>
    void EwBinaryMat<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>>::apply(BinaryOpCode opCode,
            DenseMatrix<VTres> *&res, const DenseMatrix<VTlhs> *lhs, const DenseMatrix<VTrhs> *rhs, DCTX(dctx)) {
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        int blockSize = 0;
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize = 0;

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, false, &alloc_desc);

        auto N = res->getNumItems();
        bool err = false;

        if(opCode == BinaryOpCode::ADD) {
            SumOp<VTres> op;
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, SumOp<VTres>>, 0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMat<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc), N,
                                                  op);
            }
            else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatRVec<VTres, decltype(op)>,
                                                           0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatRVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc),
                                                      numColsRhs, N, op);
            }
            else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatCVec<VTres, decltype(op)>,
                                                           0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatCVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc),
                                                      numRowsRhs, N, op);
            }
            else {
                err = true;
            }
        }
        else if(opCode == BinaryOpCode::SUB) {
            MinusOp<VTres> op;
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, decltype(op)>, 0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMat<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc), N,
                                                  op);
            }
            else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatRVec<VTres, decltype(op)>,
                                                           0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatRVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc),
                                                      numColsRhs, N, op);
            }
            else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatCVec<VTres, decltype(op)>,
                                                           0,
                                                           0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatCVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc),
                                                      numRowsRhs, N, op);
            }
            else {
                err = true;
            }
        }
        else if(opCode == BinaryOpCode::MUL) {
            ProductOp<VTres> op;
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, decltype(op)>,
                                                                0, 0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMat<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                  rhs->getValues(&alloc_desc), N, op);
            }
            else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
                CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatRVec<VTres,
                        decltype(op)>, 0, 0));
                gridSize = (N + blockSize - 1) / blockSize;

                ewBinMatRVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                      rhs->getValues(&alloc_desc), numColsRhs, N, op);
            }
            else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatCVec<VTres, decltype(op)>,
                                                           0, 0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatCVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                      rhs->getValues(&alloc_desc), numRowsRhs, N, op);
            }
            else {
                err = true;
            }
        }
        else if(opCode == BinaryOpCode::DIV) {
            DivOp<VTres> op;
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, decltype(op)>,
                                                                0, 0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMat<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                  rhs->getValues(&alloc_desc), N, op);
            }
            else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
                CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatRVec<VTres,
                        decltype(op)>, 0, 0));
                gridSize = (N + blockSize - 1) / blockSize;

                ewBinMatRVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                      rhs->getValues(&alloc_desc), numColsRhs, N, op);
            }
            else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
                CHECK_CUDART(
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMatCVec<VTres, decltype(op)>,
                                                           0, 0));
                gridSize = (N + blockSize - 1) / blockSize;
                ewBinMatCVec<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                                                      rhs->getValues(&alloc_desc), numRowsRhs, N, op);
            }
            else {
                err = true;
            }
        }

        else if (opCode == BinaryOpCode::MAX) {
            err = launch_ewbinmat<VTres, MaxOp<VTres>>(numRowsLhs, numColsLhs, numRowsRhs, numColsRhs, gridSize, minGridSize,
                blockSize, N, res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc));
        }
        else if (opCode == BinaryOpCode::NEQ) {
            err = launch_ewbinmat<VTres, NeqOp<VTres>>(numRowsLhs, numColsLhs, numRowsRhs, numColsRhs, gridSize, minGridSize,
                    blockSize, N, res->getValues(&alloc_desc), lhs->getValues(&alloc_desc), rhs->getValues(&alloc_desc));
        }
        else {
            throw std::runtime_error(fmt::format("Unknown opCode {} for EwBinaryMat", static_cast<uint32_t>(opCode)));
        }
        if(err) {
            assert(
                    false && "lhs and rhs must either have the same dimensions, "
                             "or one if them must be a row/column vector with the "
                             "width/height of the other"
            );
        }
        ctx->logger->debug("EwBinMat[{}]: {} blocks x {} threads = {} total threads for {} items",
                binary_op_codes[static_cast<int>(opCode)], gridSize, blockSize, gridSize*blockSize, N);
    }

    // ----------------------------------------------------------------------------
    // ToDo: this is not a general solution for sparse ew bin mat
    // ----------------------------------------------------------------------------
    template<typename VTres, typename VTlhs, typename VTrhs>
    void EwBinaryMat<CSRMatrix<VTres>, CSRMatrix<VTlhs>, CSRMatrix<VTrhs>>::apply(BinaryOpCode opCode,
            CSRMatrix<VTres> *&res, const CSRMatrix<VTlhs> *lhs, const CSRMatrix<VTrhs> *rhs, DCTX(dctx)) {

        if((lhs->getNumRows() != rhs->getNumRows() &&  rhs->getNumRows() != 1)
                || (lhs->getNumCols() != rhs->getNumCols() && rhs->getNumCols() != 1 ))
            throw std::runtime_error("CUDA::EwBinaryMat(CSR) - lhs and rhs must have the same dimensions (or broadcast)");

        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        // output will be n x m because of column major format of cublas
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VTres>>(lhs->getNumRows(), lhs->getNumCols(), lhs->getNumNonZeros(), false, &alloc_desc);

        if(opCode == BinaryOpCode::MUL) {
            ProductOp<VTres> op;
//            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
//            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, decltype(op)>, 0, 0));
//            gridSize = (N + blockSize - 1) / blockSize;
            auto gridSize = lhs->getNumRows();
            auto blockSize = 32;
            auto N = lhs->getNumNonZeros();

            CHECK_CUDART(cudaMemcpy(res->getRowOffsets(&alloc_desc), lhs->getRowOffsets(&alloc_desc), (lhs->getNumRows() + 1) * sizeof(size_t), cudaMemcpyDeviceToDevice));
            CHECK_CUDART(cudaMemcpy(res->getColIdxs(&alloc_desc), lhs->getColIdxs(&alloc_desc), N * sizeof(size_t), cudaMemcpyDeviceToDevice));

            ewBinMatSparseSparse<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                    res->getColIdxs(&alloc_desc), res->getRowOffsets(&alloc_desc), rhs->getValues(&alloc_desc), N, op);
//            }
        }
//        for(size_t rowIdx = 0; rowIdx < lhs->getNumRows(); rowIdx++) {
//            size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
//            if(nnzRowLhs) {
//                // intersect within row
//                auto valuesRowLhs = lhs->getValues(rowIdx, alloc_desc);
//                auto valuesRowRes = res->getValues(rowIdx, alloc_desc);
//                const size_t * colIdxsRowLhs = lhs->getColIdxs(rowIdx);
//                size_t * colIdxsRowRes = res->getColIdxs(rowIdx);
//                auto rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
//                size_t posRes = 0;
//                for (size_t posLhs = 0; posLhs < nnzRowLhs; ++posLhs) {
//                    auto rhsCol = (rhs->getNumCols() == 1 ? 0 : colIdxsRowLhs[posLhs]);
//                    auto rVal = rhs->get(rhsRow, rhsCol);
//                    if(rVal != 0) {
//                        valuesRowRes[posRes] = func(valuesRowLhs[posLhs], rVal, ctx);
//                        colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
//                        posRes++;
//                    }
//                }
//                rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
//            }
//            else
//                // empty row in result
//                rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
//        }

    }

    // ----------------------------------------------------------------------------
    // ToDo: this is not a general solution for sparse ew bin mat
    // ----------------------------------------------------------------------------
    template<typename VTres, typename VTlhs, typename VTrhs>
    void EwBinaryMat<CSRMatrix<VTres>, CSRMatrix<VTlhs>, DenseMatrix<VTrhs>>::apply(BinaryOpCode opCode,
            CSRMatrix<VTres> *&res, const CSRMatrix<VTlhs> *lhs, const DenseMatrix<VTrhs> *rhs, DCTX(dctx)) {

        if((lhs->getNumRows() != rhs->getNumRows() &&  rhs->getNumRows() != 1)
           || (lhs->getNumCols() != rhs->getNumCols() && rhs->getNumCols() != 1 ))
            throw std::runtime_error("CUDA::EwBinaryMat(CSR) - lhs and rhs must have the same dimensions (or broadcast)");

        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);

        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VTres>>(lhs->getNumRows(), lhs->getNumCols(), lhs->getNumNonZeros(), true, &alloc_desc);

        if(opCode == BinaryOpCode::MUL) {
            ProductOp<VTres> op;
//            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
//            CHECK_CUDART(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ewBinMat<VTres, decltype(op)>, 0, 0));
//            gridSize = (N + blockSize - 1) / blockSize;
            auto gridSize = lhs->getNumRows();
            auto blockSize = 32;
            auto N = lhs->getNumNonZeros();
            ewBinMatSparseDense<<<gridSize, blockSize>>>(res->getValues(&alloc_desc), lhs->getValues(&alloc_desc),
                    lhs->getColIdxs(&alloc_desc), lhs->getRowOffsets(&alloc_desc), rhs->getValues(&alloc_desc), N, op);
//            }
        }
//        for(size_t rowIdx = 0; rowIdx < lhs->getNumRows(); rowIdx++) {
//            size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
//            if(nnzRowLhs) {
//                // intersect within row
//                auto valuesRowLhs = lhs->getValues(rowIdx, alloc_desc);
//                auto valuesRowRes = res->getValues(rowIdx, alloc_desc);
//                const size_t * colIdxsRowLhs = lhs->getColIdxs(rowIdx);
//                size_t * colIdxsRowRes = res->getColIdxs(rowIdx);
//                auto rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
//                size_t posRes = 0;
//                for (size_t posLhs = 0; posLhs < nnzRowLhs; ++posLhs) {
//                    auto rhsCol = (rhs->getNumCols() == 1 ? 0 : colIdxsRowLhs[posLhs]);
//                    auto rVal = rhs->get(rhsRow, rhsCol);
//                    if(rVal != 0) {
//                        valuesRowRes[posRes] = func(valuesRowLhs[posLhs], rVal, ctx);
//                        colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
//                        posRes++;
//                    }
//                }
//                rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
//            }
//            else
//                // empty row in result
//                rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
//        }

    }

    template struct EwBinaryMat<DenseMatrix<int64_t>, DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
    template struct EwBinaryMat<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>, DenseMatrix<uint64_t>>;
    template struct EwBinaryMat<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct EwBinaryMat<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
    template struct EwBinaryMat<CSRMatrix<double>, CSRMatrix<double>, CSRMatrix<double>>;
    template struct EwBinaryMat<CSRMatrix<double>, CSRMatrix<double>, DenseMatrix<double>>;
    template struct EwBinaryMat<CSRMatrix<float>, CSRMatrix<float>, CSRMatrix<float>>;
    template struct EwBinaryMat<CSRMatrix<float>, CSRMatrix<float>, DenseMatrix<float>>;
    template struct EwBinaryMat<CSRMatrix<int64_t>, CSRMatrix<int64_t>, CSRMatrix<int64_t>>;
    template struct EwBinaryMat<CSRMatrix<int64_t>, CSRMatrix<int64_t>, DenseMatrix<int64_t>>;
}
