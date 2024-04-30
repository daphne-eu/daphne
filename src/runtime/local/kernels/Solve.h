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
#include <lapacke.h>

#include <cstddef>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct Solve {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void solve(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    Solve<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Solve<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(lhs->getNumRows());
        const auto nc1 = static_cast<int>(lhs->getNumCols());
        const auto nc2 = static_cast<int>(rhs->getNumCols());
        if (nr1 != static_cast<int>(rhs->getNumRows()))
            throw std::runtime_error(
                "#rows of lhs and #rows of rhs must be the same");
        if (nr1 != nc1)
            throw std::runtime_error("#rows and #cols of lhs must be the same");
        if (static_cast<int>(lhs->getRowSkip()) != nc1)
            throw std::runtime_error("#cols of lhs must match row skip");
        if (nc2 != 1)
            throw std::runtime_error("#cols of rhs must be 1");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);

        // solve system of equations via LU decomposition
        int ipiv[nr1];       // permutation indexes
        float work[nr1*nc1]; // LU factorization of gesv
        memcpy(work, lhs->getValues(), nr1*nc1*sizeof(float));         //for in-place A
        memcpy(res->getValues(), rhs->getValues(), nr1*sizeof(float)); //for in-place b-out
        [[maybe_unused]] int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, nr1, 1, work, nc1, ipiv, res->getValues(), 1);
        if (info > 0)
            throw std::runtime_error("A factor Ui is exactly singular, so the solution could not be computed");
    }
};

template<>
struct Solve<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(lhs->getNumRows());
        const auto nc1 = static_cast<int>(lhs->getNumCols());
        const auto nc2 = static_cast<int>(rhs->getNumCols());
        if (nr1 != static_cast<int>(rhs->getNumRows()))
            throw std::runtime_error(
                "#rows of lhs and #rows of rhs must be the same");
        if (nr1 != nc1)
            throw std::runtime_error("#rows and #cols of lhs must be the same");
        if (static_cast<int>(lhs->getRowSkip()) != nc1)
            throw std::runtime_error("#cols of lhs must match row skip");
        if (nc2 != 1)
            throw std::runtime_error("#cols of rhs must be 1");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        // solve system of equations via LU decomposition
        int ipiv[nr1];       // permutation indexes
        double work[nr1*nc1]; // LU factorization of gesv
        memcpy(work, lhs->getValues(), nr1*nc1*sizeof(double));         //for in-place A
        memcpy(res->getValues(), rhs->getValues(), nr1*sizeof(double)); //for in-place b-out
        [[maybe_unused]] int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, nr1, 1, work, nc1, ipiv, res->getValues(), 1);
        if (info > 0)
            throw std::runtime_error("A factor Ui is exactly singular, so the solution could not be computed");
    }
};
