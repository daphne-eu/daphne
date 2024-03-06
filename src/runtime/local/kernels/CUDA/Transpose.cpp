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

#include "Transpose.h"

namespace CUDA {
    template<>
    [[maybe_unused]] void launch_cublas_geam<double>(const CUDAContext &ctx, size_t m, size_t n, const double *alpha,
                                                     const double *beta, const double *A, double *C) {
        auto lda = n;
        auto ldb = n;
        auto ldc = m;
        CHECK_CUBLAS(cublasDgeam(ctx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, A,
                                 ldb, C, ldc));
    }

    template<>
    [[maybe_unused]] void launch_cublas_geam<float>(const CUDAContext &ctx, size_t m, size_t n, const float *alpha,
                                                    const float *beta, const float *A, float *C) {
        auto lda = n;
        auto ldb = n;
        auto ldc = m;
        CHECK_CUBLAS(cublasSgeam(ctx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, A,
                                 ldb, C, ldc));
    }

    template<>
    [[maybe_unused]] void launch_cublas_geam<int64_t>(const CUDAContext &ctx, size_t m, size_t n, const int64_t *alpha,
                                                      const int64_t *beta, const int64_t *A, int64_t *C) {
        auto lda = n;
        auto ldb = n;
        auto ldc = m;
        auto alpha_ = static_cast<double>(*alpha);
        auto beta_ = static_cast<double>(*beta);
        CHECK_CUBLAS(cublasDgeam(ctx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha_,
                                 reinterpret_cast<const double *>(A), lda, &beta_, reinterpret_cast<const double *>(A),
                                 ldb, reinterpret_cast<double *>(C), ldc));
    }

    template<>
    [[maybe_unused]] void launch_cublas_geam<uint64_t>(const CUDAContext &ctx, size_t m, size_t n, const uint64_t *alpha,
                                                      const uint64_t *beta, const uint64_t *A, uint64_t *C) {
        auto lda = n;
        auto ldb = n;
        auto ldc = m;
        auto alpha_ = static_cast<double>(*alpha);
        auto beta_ = static_cast<double>(*beta);
        CHECK_CUBLAS(cublasDgeam(ctx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha_,
                                 reinterpret_cast<const double *>(A), lda, &beta_, reinterpret_cast<const double *>(A),
                                 ldb, reinterpret_cast<double *>(C), ldc));

    }

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------
    
    template<typename VT>
    void Transpose<DenseMatrix<VT>, DenseMatrix<VT>>::apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg,
            DCTX(dctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;
        const VT *d_arg = arg->getValues(&alloc_desc);
        
        // skip data movement for vectors
        if ((numRows == 1 || numCols == 1) && !arg->isView()) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, arg);
        }
        else
        {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, false, &alloc_desc);
            VT *d_res = res->getValues(&alloc_desc);
            launch_cublas_geam<VT>(*ctx, numRows, numCols, &blend_alpha, &blend_beta, d_arg, d_res);
        }
    }
    template struct Transpose<DenseMatrix<double>, DenseMatrix<double>>;
    template struct Transpose<DenseMatrix<float>, DenseMatrix<float>>;
    template struct Transpose<DenseMatrix<int64_t>, DenseMatrix<int64_t>>;
    template struct Transpose<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>>;
}