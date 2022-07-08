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

#include "runtime/local/datastructures/AllocationDescriptorCUDA.h"
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>

namespace CUDA {
    template<typename T>
    void launch_cublas_geam(const CUDAContext &ctx, size_t m, size_t n, const T *alpha, const T *beta, const T *A, T *C);


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

    template<class DTRes, class DTArg>
    struct Transpose {
        static void apply(DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
    };

    template<typename VT>
    struct Transpose<DenseMatrix<VT>, DenseMatrix<VT>> {
        static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx));
    };

// ****************************************************************************
// Convenience function
// ****************************************************************************

    template<class DTRes, class DTArg>
    void transpose(DTRes *&res, const DTArg *arg, DCTX(ctx)) {
        Transpose<DTRes, DTArg>::apply(res, arg, ctx);
    }
}