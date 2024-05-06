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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_COLBIND_H
#define SRC_RUNTIME_LOCAL_KERNELS_COLBIND_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct ColBind {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void colBind(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    ColBind<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct ColBind<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();

        if (numRows != rhs->getNumRows()) {
            throw std::runtime_error(
                "ColBind - lhs and rhs must have the same number of rows");
        }

        const size_t numColsLhs = lhs->getNumCols();
        const size_t numColsRhs = rhs->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsLhs + numColsRhs, false);
        
        const VT * valuesLhs = lhs->getValues();
        const VT * valuesRhs = rhs->getValues();
        VT * valuesRes = res->getValues();
        
        const size_t rowSkipLhs = lhs->getRowSkip();
        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();
        
        for(size_t r = 0; r < numRows; r++) {
            memcpy(valuesRes             , valuesLhs, numColsLhs * sizeof(VT));
            memcpy(valuesRes + numColsLhs, valuesRhs, numColsRhs * sizeof(VT));
            valuesLhs += rowSkipLhs;
            valuesRhs += rowSkipRhs;
            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, Frame
// ----------------------------------------------------------------------------

template<>
struct ColBind<Frame, Frame, Frame> {
    static void apply(Frame *& res, const Frame * lhs, const Frame * rhs, DCTX(ctx)) {
        res = DataObjectFactory::create<Frame>(lhs, rhs);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct ColBind<CSRMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, DCTX(ctx)) {
        if(lhs->getNumRows() != rhs->getNumRows())
            throw std::runtime_error("lhs and rhs must have the same number of rows");

        size_t numColsRes = lhs->getNumCols() + rhs->getNumCols();
        size_t numNonZerosRes = lhs->getNumNonZeros() + rhs->getNumNonZeros();

        if(!res)
            res = DataObjectFactory::create<CSRMatrix<VT>>(lhs->getNumRows(), numColsRes, numNonZerosRes, false);

        auto resRowOffsets = res->getRowOffsets();
        auto lhsRowOffsets = lhs->getRowOffsets();
        auto rhsRowOffsets = rhs->getRowOffsets();

        size_t rhsPrevEnd = 0;
        size_t lhsStartOffset = lhsRowOffsets[0];
        size_t rhsStartOffset = rhsRowOffsets[0];

        for(size_t r = 0; r < res->getNumRows(); r++){
            size_t lhsOffset = rhsPrevEnd;
            size_t lhsLength = lhsRowOffsets[r + 1] - lhsRowOffsets[r];
            size_t rhsOffset = lhsOffset + lhsLength;
            size_t rhsLength = rhsRowOffsets[r + 1] - rhsRowOffsets[r];
            rhsPrevEnd = rhsOffset + rhsLength;

            memcpy(&res->getValues()[lhsOffset], &lhs->getValues()[lhsRowOffsets[r]], lhsLength * sizeof(VT));
            memcpy(&res->getValues()[rhsOffset], &rhs->getValues()[rhsRowOffsets[r]], rhsLength * sizeof(VT));

            memcpy(&res->getColIdxs()[lhsOffset], &lhs->getColIdxs()[lhsRowOffsets[r]], lhsLength * sizeof(size_t));
            for(size_t c = 0; c < rhsLength; c++)
                res->getColIdxs()[rhsOffset + c] = rhs->getColIdxs()[rhsRowOffsets[r] + c] + lhs->getNumCols();

            resRowOffsets[r] = lhsRowOffsets[r] - lhsStartOffset + rhsRowOffsets[r] - rhsStartOffset;
        }
        resRowOffsets[res->getNumRows()] = numNonZerosRes;
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_COLBIND_H
