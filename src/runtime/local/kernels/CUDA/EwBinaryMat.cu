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
#include <cstdint>

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

    template<class T>
    void launch_axpy(const cublasHandle_t &handle, int n, const T *alpha, const T *x, int incx,
                     T *y, int incy);

    template<>
    void launch_axpy<double>(const cublasHandle_t &handle, int n, const double *alpha, const double *x, int incx,
                             double *y, int incy) {
        CHECK_CUBLAS(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
    }

    template<>
    void launch_axpy<float>(const cublasHandle_t &handle, int n, const float *alpha, const float *x, int incx,
                            float *y, int incy) {
        CHECK_CUBLAS(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
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

        int blockSize;
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        size_t gridSize;

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, false, &alloc_desc);

        auto N = res->getNumItems();
        bool err = false;

        if(opCode == BinaryOpCode::ADD) {
            SumOp<VTres> op;
            if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
                // auto ctx = dctx->getCUDAContext(0);

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
        spdlog::get("runtime::cuda")->debug("EwBinMat[{}]: {} blocks x {} threads = {} total threads for {} items",
                static_cast<int>(opCode), gridSize, blockSize, gridSize*blockSize,N);
    }

    template struct EwBinaryMat<DenseMatrix<long>, DenseMatrix<long>, DenseMatrix<long>>;
    template struct EwBinaryMat<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct EwBinaryMat<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
}
