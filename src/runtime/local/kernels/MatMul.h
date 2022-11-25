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
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cblas.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool transa, bool transb, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, transa, transb, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(transa ? lhs->getNumCols() : lhs->getNumRows());
        const auto nc1 = static_cast<int>(transa ? lhs->getNumRows() : lhs->getNumCols());
        const auto nc2 = static_cast<int>(transb ? rhs->getNumRows() : rhs->getNumCols());
        assert((nc1 == static_cast<int>(transb ? rhs->getNumCols() : rhs->getNumRows())) && "#cols of lhs and #rows of rhs must be the same");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            res->set(0, 0, cblas_sdot(nc1, lhs->getValues(), 1, rhs->getValues(),
                    static_cast<int>(rhs->getRowSkip())));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_sgemv(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, lhs->getNumRows(), lhs->getNumCols(), 1, lhs->getValues(),
                static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                static_cast<int>(rhs->getRowSkip()), 0, res->getValues(),
                static_cast<int>(res->getRowSkip()));
        else                     // Matrix-Matrix
            cblas_sgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans, nr1, nc2, nc1,
                1, lhs->getValues(), static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                static_cast<int>(rhs->getRowSkip()), 0, res->getValues(), static_cast<int>(res->getRowSkip()));
    }
};

template<>
struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs, bool transa, bool transb, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(transa ? lhs->getNumCols() : lhs->getNumRows());
        const auto nc1 = static_cast<int>(transa ? lhs->getNumRows() : lhs->getNumCols());
        const auto nc2 = static_cast<int>(transb ? rhs->getNumRows() : rhs->getNumCols());
        assert((nc1 == static_cast<int>(transb ? rhs->getNumCols() : rhs->getNumRows())) && "#cols of lhs and #rows of rhs must be the same");
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            res->set(0, 0, cblas_ddot(nc1, lhs->getValues(), 1, rhs->getValues(),
                static_cast<int>(rhs->getRowSkip())));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_dgemv(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, lhs->getNumRows(), lhs->getNumCols(), 1, lhs->getValues(),
                static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                static_cast<int>(rhs->getRowSkip()), 0, res->getValues(),
                static_cast<int>(res->getRowSkip()));
        else                     // Matrix-Matrix
            cblas_dgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans, transb ? CblasTrans : CblasNoTrans, nr1, nc2, nc1,
                1, lhs->getValues(), static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                static_cast<int>(rhs->getRowSkip()), 0, res->getValues(), static_cast<int>(res->getRowSkip()));
    }
};
