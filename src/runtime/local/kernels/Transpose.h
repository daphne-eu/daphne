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

template<class DTRes, class DTArg>
struct Transpose {
    static void apply(DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void transpose(DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    Transpose<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Transpose<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        // skip data movement for vectors
        if ((numRows == 1 || numCols == 1) && !arg->isView()) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, arg);
        }
        else
        {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(numCols, numRows, false);

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

template<typename VT>
struct Transpose<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numCols, numRows, arg->getNumNonZeros(), false);
        
        const VT * valuesArg = arg->getValues();
        const size_t * colIdxsArg = arg->getColIdxs();
        const size_t * rowOffsetsArg = arg->getRowOffsets();
        
        VT * valuesRes = res->getValues();
        VT * const valuesResInit = valuesRes;
        size_t * colIdxsRes = res->getColIdxs();
        size_t * rowOffsetsRes = res->getRowOffsets();
        
        auto* curRowOffsets = new size_t[numRows + 1];
        memcpy(curRowOffsets, rowOffsetsArg, (numRows + 1) * sizeof(size_t));
        
        rowOffsetsRes[0] = 0;
        for(size_t c = 0; c < numCols; c++) {
            for(size_t r = 0; r < numRows; r++)
                if(curRowOffsets[r] < rowOffsetsArg[r + 1] && colIdxsArg[curRowOffsets[r]] == c) {
                    *valuesRes++ = valuesArg[curRowOffsets[r]];
                    *colIdxsRes++ = r;
                    curRowOffsets[r]++;
                }
            rowOffsetsRes[c + 1] = valuesRes - valuesResInit;
        }
        
        delete[] curRowOffsets;
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct Transpose<Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *& res, const Matrix<VT> * arg, DCTX(ctx)) {
        const size_t numRowsRes = arg->getNumCols();
        const size_t numColsRes = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numColsRes, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRowsRes; ++r)
            for (size_t c = 0; c < numColsRes; ++c)
                res->append(r, c, arg->get(c, r));
        res->finishAppend();
    }
};