/*
 * Copyright 2025 The DAPHNE Consortium
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

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct ConvertBitmapToPosList {
    static void apply(DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg> void convertBitmapToPosList(DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    ConvertBitmapToPosList<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct ConvertBitmapToPosList<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numColsArg = arg->getNumCols();
        if (numColsArg != 1)
            throw std::runtime_error("the argument must have exactly one column but has " + std::to_string(numColsArg) +
                                     " columns");

        const size_t numRowsArg = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRowsArg, 1, false);

        const VTArg *valuesArg = arg->getValues();
        VTRes *valuesRes = res->getValues();
        size_t numRowsRes = 0;

        for (size_t r = 0; r < numRowsArg; r++) {
            if (*valuesArg == 1) {
                *valuesRes = r;
                valuesRes += res->getRowSkip();
                numRowsRes++;
            }
            valuesArg += arg->getRowSkip();
        }

        res->shrinkNumRows(numRowsRes);
    }
};