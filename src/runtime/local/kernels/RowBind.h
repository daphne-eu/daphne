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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H
#define SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct RowBind {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void rowBind(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    RowBind<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RowBind<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numCols = lhs->getNumCols();
        assert((numCols == rhs->getNumCols()) && "lhs and rhs must have the same number of cols");

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numRowsRhs = rhs->getNumRows();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numColsLhs + numColsRhs, false);

        const VT * valuesLhs = lhs->getValues();
        const VT * valuesRhs = rhs->getValues();
        VT * valuesRes = res->getValues();
        
        const size_t colSkipLhs = lhs->getColSkip();
        const size_t colSkipRhs = rhs->getColSkip();
        const size_t colSkipRes = res->getColSkip();

        for(size_t r = 0; r < numCols; r++) {
            memcpy(valuesRes             , valuesLhs, numRowsLhs * sizeof(VT));
            memcpy(valuesRes + numRowsLhs, valuesRhs, numRowsRhs * sizeof(VT));
            valuesLhs += colSkipLhs;
            valuesRhs += colSkipRhs;
            valuesRes += colSkipRes;
        }

    }
};


template<>
struct RowBind<Frame, Frame, Frame> {
    static void apply(Frame *& res, const Frame * lhs, const Frame * rhs, DCTX(ctx)) {
        res = DataObjectFactory::create<Frame>(lhs, rhs);
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_ROWBIND_H


