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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <random>
#include <set>
#include <type_traits>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace CUDA {
// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

    template<class DTRes, typename VTArg>
    struct RandMatrix {
        static void
        apply(DTRes *&res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed,
              DCTX(ctx)) = delete;
    };

// ****************************************************************************
// Convenience function
// ****************************************************************************

    template<class DTRes, typename VTArg>
    void randMatrix(DTRes *&res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed,
                    DCTX(ctx)) {
        RandMatrix<DTRes, VTArg>::apply(res, numRows, numCols, min, max, sparsity, seed, ctx);
    }

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

    template<typename VT>
    struct RandMatrix<DenseMatrix<VT>, VT> {
        static void apply(DenseMatrix<VT> *&res, size_t numRows, size_t numCols, VT min, VT max, double sparsity,
            int64_t seed, DCTX(ctx));
    };

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

    template<typename VT>
    struct RandMatrix<CSRMatrix<VT>, VT> {
        static void
        apply(CSRMatrix<VT> *&res, size_t numRows, size_t numCols, VT min, VT max, double sparsity, int64_t seed,
              DCTX(ctx)) {

        }
    };
}