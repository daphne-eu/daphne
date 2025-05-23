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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, typename VTArg> struct Fill {
    static void apply(DTRes *&res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, typename VTArg> void fill(DTRes *&res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) {
    Fill<DTRes, VTArg>::apply(res, arg, numRows, numCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct Fill<DenseMatrix<VTRes>, VTArg> {
    static void apply(DenseMatrix<VTRes> *&res, VTArg arg, size_t numRows, size_t numCols, DCTX(ctx)) {
        if (res != nullptr)
            throw std::invalid_argument("Trying to fill an already existing DenseMatrix.");

        res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
        std::fill(res->getValues(), res->getValues() + res->getNumItems(), arg);
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct Fill<Matrix<VT>, VT> {
    static void apply(Matrix<VT> *&res, VT arg, size_t numRows, size_t numCols, DCTX(ctx)) {
        if (res != nullptr)
            throw std::invalid_argument("Trying to fill an already existing DenseMatrix.");

        res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, arg == 0);
        if (arg != 0) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r)
                for (size_t c = 0; c < numCols; ++c)
                    res->append(r, c, arg);
            res->finishAppend();
        }
    }
};
