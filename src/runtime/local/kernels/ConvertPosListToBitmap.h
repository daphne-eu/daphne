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

template <class DTRes, class DTArg> struct ConvertPosListToBitmap {
    static void apply(DTRes *&res, const DTArg *arg, const size_t numRowsRes, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void convertPosListToBitmap(DTRes *&res, const DTArg *arg, size_t numRowsRes, DCTX(ctx)) {
    ConvertPosListToBitmap<DTRes, DTArg>::apply(res, arg, numRowsRes, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct ConvertPosListToBitmap<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, size_t numRowsRes, DCTX(ctx)) {
        const size_t numColsArg = arg->getNumCols();
        if (numColsArg != 1)
            throw std::runtime_error("the argument must have exactly one column but has " + std::to_string(numColsArg) +
                                     " columns");

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRowsRes, 1, true);

        const VTArg *valuesArg = arg->getValues();
        VTRes *valuesRes = res->getValues();

        for (size_t r = 0; r < arg->getNumRows(); r++) {
            const size_t pos = *valuesArg;
            if (pos > numRowsRes)
                throw std::runtime_error("out-of-bounds access: trying to set position " + std::to_string(pos) +
                                         " in a column matrix with " + std::to_string(numRowsRes) + " rows");
            valuesRes[pos * res->getRowSkip()] = 1;
            valuesArg += arg->getRowSkip();
        }
    }
};