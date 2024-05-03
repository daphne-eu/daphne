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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H
#define SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <algorithm>
#include <type_traits>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename DTArg>
struct Reverse {
    static void apply(DTRes *& res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename DTArg>
void reverse(DTRes *& res, const DTArg *arg, DCTX(ctx)) {
    Reverse<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Reverse<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        size_t numRows = arg->getNumRows();
        size_t numCols = arg->getNumCols();
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        const VT *valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        // This operation will often be applied to column (n x 1) matrices,
        // so this case could optionally be treated more efficiently.
        if (arg->getRowSkip() == 1){ // We need to check RowSkip in case of sub Matrix (see DenseMatrix.h)
            std::reverse_copy(valuesArg, valuesArg + numRows, valuesRes);
        }
        else {
            const VT *valuesArgLastRow = valuesArg + ((numRows - 1) * arg->getRowSkip());
            for (size_t r = 0; r < numRows; r++) {
                memcpy(valuesRes, valuesArgLastRow, numCols * sizeof(VT));
                valuesRes += res->getRowSkip();
                valuesArgLastRow -= arg->getRowSkip();
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Reverse<Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, arg->get((numRows - 1) - r, c));
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_REVERSE_H
