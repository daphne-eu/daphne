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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CastObj.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) = delete;
};

template<typename T>
struct MatMul<DenseMatrix<T>, DenseMatrix<T>, DenseMatrix<T>> {
    static void apply(DenseMatrix<T> *&res, const DenseMatrix<T> *lhs, const DenseMatrix<T> *rhs, bool transa,
                      bool transb, DCTX(dctx));
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, transa, transb, ctx);
}

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct MatMul<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        [[maybe_unused]] const size_t nc1 = lhs->getNumCols();

        [[maybe_unused]] const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();

        if (nc1 != nr2) {
            throw std::runtime_error(
                "MatMul - #cols of lhs and #rows of rhs must be the same");
        }
        // FIXME: transpose isn't supported atm

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(nr1, nc2, false);

        const VT * valuesRhs = rhs->getValues();
        VT * valuesRes = res->getValues();

        const size_t rowSkipRhs = rhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        memset(valuesRes, VT(0), sizeof(VT) * nr1 * nc2);
        for(size_t r = 0; r < nr1; r++) {
            const size_t rowNumNonZeros = lhs->getNumNonZeros(r);
            const size_t * rowColIdxs = lhs->getColIdxs(r);
            const VT * rowValues = lhs->getValues(r);

            const size_t rowIdxRes = r * rowSkipRes;
            for(size_t i = 0; i < rowNumNonZeros; i++) {
                const size_t c = rowColIdxs[i];
                const size_t rowIdxRhs = c * rowSkipRhs;

                for(size_t j = 0; j < nc2; j++) {
		            valuesRes[rowIdxRes + j] += rowValues[i] * valuesRhs[rowIdxRhs + j];
                }
            }
        }
    }
};



