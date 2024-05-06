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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_TRI_H
#define SRC_RUNTIME_LOCAL_KERNELS_TRI_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>
#include <stdexcept>
#include <stdio.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct Tri {
    /**
     * @brief lower/upperTri
     * @param res Result pointer
     * @param arg Input pointer
     * @param upper Selects between upperTri (true) and lowerTri (false)
     * @param diag Preserves (true) or zeroes (false) the diagonal
     * @param values Preserves (true) or replaces with 1s the remaining elements
     */
    static void apply(DT *& res, const DT * arg, bool upper, bool diag, bool values, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DT>
void tri(DT *& res, const DT * arg, bool upper, bool diag, bool values, DCTX(ctx)) {
    Tri<DT>::apply(res, arg, upper, diag, values, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Tri<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, bool upper, bool diag, bool values, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (numRows != numCols) {
            throw std::runtime_error("matrix must be square, but is of shape" +
                                     std::to_string(numRows) + "x" +
                                     std::to_string(numCols));
        }

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, true);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        const size_t rowSkipArg = arg->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        size_t start = upper ? !diag : 0;
        size_t end = upper ? numCols : diag;
        size_t * inc = upper ? &start : &end;

        for(size_t r = 0; r < numRows; r++, (*inc)++) {
            for(size_t c = start; c < end; c++) {
                VT val = valuesArg[c];
                if(val != VT(0)) {
                    valuesRes[c] = !values ? 1 : val;
                }
            }
            valuesArg += rowSkipArg;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Tri<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, bool upper, bool diag, bool values, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (numRows != numCols) {
            throw std::runtime_error("matrix must be square, but is of shape" +
                                     std::to_string(numRows) + "x" +
                                     std::to_string(numCols));
        }
        if(res == nullptr) {
            const size_t nonZeros = std::min(arg->getNumNonZeros(), numRows * (numRows + 1) / 2);
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, nonZeros, false);
        }

        size_t start = upper ? !diag : 0;
        size_t end = upper ? numCols : diag;
        size_t * inc = upper ? &start : &end;

        VT * valuesRes = res->getValues();
        size_t * colIdxsRes = res->getColIdxs();
        size_t * rowOffsetsRes = res->getRowOffsets();

        rowOffsetsRes[0] = 0;
        for(size_t r = 0, pos = 0; r < numRows; r++, (*inc)++) {
            const size_t rowNumNonZeros = arg->getNumNonZeros(r);
            const size_t * rowColIdxs = arg->getColIdxs(r);
            const VT * rowValues = arg->getValues(r);

            for(size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                if(c >= start && c < end) {
                    VT val = rowValues[i];
                    if(val != VT(0)) {
                        valuesRes[pos] = !values ? 1 : val;
                        colIdxsRes[pos++] = c;
                    }
                }
            }
            rowOffsetsRes[r + 1] = pos;
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_TRI_H
