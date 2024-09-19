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
#include <runtime/local/datastructures/Matrix.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct Transpose {
    static void apply(DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void transpose(DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    Transpose<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Transpose<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg,
                      DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        // skip data movement for vectors
        if ((numRows == 1 || numCols == 1) && !arg->isView()) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows,
                                                             arg);
        } else {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(
                    numCols, numRows, false);

            const VT *valuesArg = arg->getValues();
            const size_t rowSkipArg = arg->getRowSkip();
            const size_t rowSkipRes = res->getRowSkip();
            for (size_t r = 0; r < numRows; r++) {
                VT *valuesRes = res->getValues() + r;
                for (size_t c = 0; c < numCols; c++) {
                    *valuesRes = valuesArg[c];
                    valuesRes += rowSkipRes;
                }
                valuesArg += rowSkipArg;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Transpose<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const CSRMatrix<VT> *arg,
                      DCTX(ctx)) {
        // Implementation inspired by SciPy
        // https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h#L608
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(
                numCols, numRows, arg->getNumNonZeros(), false);

        const VT *valuesArg = arg->getValues();
        const size_t *colIdxsArg = arg->getColIdxs();
        const size_t *rowOffsetsArg = arg->getRowOffsets();

        const size_t numNonZeros = arg->getNumNonZeros();

        VT *valuesRes = res->getValues();
        size_t *colIdxsRes = res->getColIdxs();
        size_t *rowOffsetsRes = res->getRowOffsets();

        std::fill(rowOffsetsRes, rowOffsetsRes + numCols, 0);

        for (size_t row = 0; row < numRows; row++)
            for (size_t j = rowOffsetsArg[row]; j < rowOffsetsArg[row + 1]; j++)
                rowOffsetsRes[colIdxsArg[j]]++;

        for (size_t col = 0, cumsum = 0; col < numCols; col++) {
            size_t tmp = rowOffsetsRes[col];
            rowOffsetsRes[col] = cumsum;
            cumsum += tmp;
        }
        rowOffsetsRes[numCols] = numNonZeros;

        for (size_t row = 0; row < numRows; row++) {
            for (size_t j = rowOffsetsArg[row]; j < rowOffsetsArg[row + 1];
                 j++) {
                size_t col = colIdxsArg[j];
                size_t dest = rowOffsetsRes[col];
                colIdxsRes[dest] = row;
                valuesRes[dest] = valuesArg[j];
                rowOffsetsRes[col]++;
            }
        }

        for (size_t col = 0, last = 0; col < numCols + 1; col++) {
            size_t tmp = rowOffsetsRes[col];
            rowOffsetsRes[col] = last;
            last = tmp;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct Transpose<Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *&res, const Matrix<VT> *arg, DCTX(ctx)) {
        const size_t numRowsRes = arg->getNumCols();
        const size_t numColsRes = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes,
                                                             numColsRes, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRowsRes; ++r)
            for (size_t c = 0; c < numColsRes; ++c)
                res->append(r, c, arg->get(c, r));
        res->finishAppend();
    }
};
