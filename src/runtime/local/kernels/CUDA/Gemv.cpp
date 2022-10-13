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

#include "Gemv.h"
#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"

namespace CUDA {
    template<>
    [[maybe_unused]] void
    launch_cublas_gemv<double>(const CUDAContext &ctx, size_t m, size_t n, const double *alpha, const double *beta,
                               const double *A, const double *x, double *y, cublasOperation_t opA) {
        // fixed for row major format
        CHECK_CUBLAS(cublasDgemv(ctx.getCublasHandle(), opA, m, n, alpha, A, m, x, 1, beta, y, 1));
    }

    template<>
    [[maybe_unused]] void launch_cublas_gemv<float>(const CUDAContext &ctx, size_t m, size_t n, const float *alpha,
            const float *beta, const float *A, const float *x, float *y, cublasOperation_t opA) {
        // fixed for row major format
        CHECK_CUBLAS(cublasSgemv(ctx.getCublasHandle(), opA, m, n, alpha, A, m, x, 1, beta, y, 1));
    }

    template<typename T>
    void Gemv<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>>::apply(DenseMatrix<T> *&res, const DenseMatrix<T> *mat,
                                                                const DenseMatrix<T> *vec, DCTX(dctx)) {

        using VT = typename DenseMatrix<T>::VT;
        const size_t deviceID = 0; //ToDo: multi device support
        auto ctx = CUDAContext::get(dctx, deviceID);
        AllocationDescriptorCUDA alloc_desc(dctx, deviceID);
        
        const size_t numRows = mat->getNumRows();
        const size_t numCols = mat->getNumCols();
        const VT blend_alpha = 1.0f;
        const VT blend_beta = 0.0f;
        const VT *d_mat = mat->getValues(&alloc_desc);
        const VT *d_vec = vec->getValues(&alloc_desc);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<T>>(numCols, 1, false, &alloc_desc);
        VT *d_res = res->getValues(&alloc_desc);

//        launch_cublas_gemv<VT>(*ctx, numRows, numCols, &blend_alpha, &blend_beta, d_mat, d_vec, d_res);
// Note: This invocation is supposed to be transposed(needed for lm microbench) + fixed for col-major behavior of cublas
        launch_cublas_gemv<VT>(*ctx, numCols, numRows, &blend_alpha, &blend_beta, d_mat, d_vec, d_res, CUBLAS_OP_N);
    }

    // explicit instantiations to satisfy linker
    template struct Gemv<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>>;
    template struct Gemv<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>;
}