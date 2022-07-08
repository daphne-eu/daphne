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

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CUDA/HostUtils.h>
#include <runtime/local/context/CUDAContext.h>

namespace CUDA {

    template<typename T>
    void launch_cublas_gemv(const CUDAContext& ctx, size_t m, size_t n, const T* alpha, const T* beta, const T* A,
                            const T* x, T* y, cublasOperation_t opA);

    // ****************************************************************************
    // Struct for partial template specialization
    // ****************************************************************************
    template<class DTRes, class DTMat, class DTVec>
    struct Gemv {
        static void apply(DTRes*& res, const DTMat* mat, const DTVec* vec, DCTX(dctx)) = delete;
    };

    template<typename T>
    struct Gemv<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>> {
        static void apply(DenseMatrix<T>*& res, const DenseMatrix<T>* mat, const DenseMatrix<T>* vec, DCTX(dctx));
    };

    // ****************************************************************************
    // Convenience function
    // ****************************************************************************
    template<class DTRes, class DTMat, class DTVec>
    void gemv(DTRes*& res, const DTMat* mat, const DTVec* vec, DCTX(ctx)) {
        Gemv<DTRes, DTMat, DTVec>::apply(res, mat, vec, ctx);
    }
}