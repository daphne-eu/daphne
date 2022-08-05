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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H
#define SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct CTable {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void ctable(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    CTable<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CTable<DenseMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctable: lhs and rhs must have only one column");
        if(lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctable: lhs and rhs must have the same number of rows");
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(*std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1, 
                                                                *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1, true);

        // res[i, j] = |{ k | lhs[k] = i and rhs[k] = j, 0 ≤ k ≤ n-1 }|.
        auto resVals = res->getValues();
        const size_t resRowSkip = res->getRowSkip();
        for(size_t c = 0; c < lhsNumRows; c++)
            resVals[static_cast<size_t>(lhsVals[c] * resRowSkip + rhsVals[c])]++;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------
template<typename VT>
struct CTable<CSRMatrix<VT>, DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(CSRMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctable: lhs and rhs must have only one column");
        if(lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctable: lhs and rhs must have the same number of rows");
        if(res == nullptr) {
            const size_t resNumRows = *std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1;
            const size_t resNumCols = *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1;
            res = DataObjectFactory::create<CSRMatrix<VT>>(resNumRows, resNumCols, std::min(lhsNumRows, resNumRows * resNumCols), true);
        }

        for(size_t c = 0; c < lhsNumRows; c++){
            auto i = lhsVals[c];
            auto j = rhsVals[c];
            res->set(i, j, res->get(i, j) + 1);
        }
    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H
