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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SOLVE_H
#define SRC_RUNTIME_LOCAL_KERNELS_SOLVE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cblas.h>
#include <lapacke.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct Solve {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool triangLhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void solve(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, bool triangLhs, DCTX(ctx)) {
    Solve<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, triangLhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct Solve<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs, bool triangLhs, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();
        assert((nr1 == nr2) && "#rows of lhs and #rows of rhs must be the same");
        assert((nr1 == nc1) && "#rows and #cols of lhs must be the same");
        assert((lhs->getRowSkip() == nc1) && "#cols of lhs must match row skip");
        assert((nc2==1) && "#cols of rhs must be 1");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);

        // solve system of equations via LU decomposition
        int ipiv[nr1];       // permutation indexes
        float work[nr1*nc1]; // LU factorization of gesv
        memcpy(work, lhs->getValues(), nr1*nc1*sizeof(float));         //for in-place A
        memcpy(res->getValues(), rhs->getValues(), nr1*sizeof(float)); //for in-place b-out
        int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, nr1, 1, work, nc1, ipiv, res->getValues(), 1);
        assert((info<=0) && "A factor Ui is exactly singular, so the solution could not be computed");
    }
};

template<>
struct Solve<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs, bool triangLhs, DCTX(ctx)) {
        const size_t nr1 = lhs->getNumRows();
        const size_t nc1 = lhs->getNumCols();
        const size_t nr2 = rhs->getNumRows();
        const size_t nc2 = rhs->getNumCols();
        assert((nr1 == nr2) && "#rows of lhs and #rows of rhs must be the same");
        assert((nr1 == nc1) && "#rows and #cols of lhs must be the same");
        assert((lhs->getRowSkip() == nc1) && "#cols of lhs must match row skip");
        assert((nc2==1) && "#cols of rhs must be 1");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        // solve system of equations via LU decomposition
        int ipiv[nr1];       // permutation indexes
        double work[nr1*nc1]; // LU factorization of gesv
        memcpy(work, lhs->getValues(), nr1*nc1*sizeof(double));         //for in-place A
        memcpy(res->getValues(), rhs->getValues(), nr1*sizeof(double)); //for in-place b-out
        int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, nr1, 1, work, nc1, ipiv, res->getValues(), 1);
        assert((info<=0) && "A factor Ui is exactly singular, so the solution could not be computed");
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_SOLVE_H
