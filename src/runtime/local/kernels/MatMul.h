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
#include <runtime/local/kernels/CastObj.h>

#include <cblas.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct MatMul {
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(lhs->getNumRows());
        const auto nc1 = static_cast<int>(lhs->getNumCols());
        const auto nc2 = static_cast<int>(rhs->getNumCols());
        assert((nc1 == static_cast<int>(rhs->getNumRows())) && "#cols of lhs and #rows of rhs must be the same");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            res->set(0, 0, cblas_sdot(nc1, lhs->getValues(), 1, rhs->getValues(),
                    static_cast<int>(rhs->getRowSkip())));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_sgemv(CblasRowMajor, CblasNoTrans, nr1, nc1, 1, lhs->getValues(),
                    static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                    static_cast<int>(rhs->getRowSkip()), 0,res->getValues(),
                    static_cast<int>(res->getRowSkip()));
        else                     // Matrix-Matrix
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr1, nc2, nc1,
                    1, lhs->getValues(), static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                static_cast<int>(rhs->getRowSkip()), 0, res->getValues(), static_cast<int>(res->getRowSkip()));
    }
};

template<>
struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(lhs->getNumRows());
        const auto nc1 = static_cast<int>(lhs->getNumCols());
        const auto nc2 = static_cast<int>(rhs->getNumCols());
        assert((nc1 == static_cast<int>(rhs->getNumRows())) && "#cols of lhs and #rows of rhs must be the same");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            res->set(0, 0, cblas_ddot(nc1, lhs->getValues(), 1, rhs->getValues(),
                static_cast<int>(rhs->getRowSkip())));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nr1, nc1, 1, lhs->getValues(),
            static_cast<int>(lhs->getRowSkip()), rhs->getValues(),static_cast<int>(rhs->getRowSkip()), 0,
                res->getValues(), static_cast<int>(res->getRowSkip()));
        else                     // Matrix-Matrix
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr1, nc2, nc1,
                    1, lhs->getValues(), static_cast<int>(lhs->getRowSkip()), rhs->getValues(),
                    static_cast<int>(rhs->getRowSkip()), 0, res->getValues(), static_cast<int>(res->getRowSkip()));
    }
};


template<>
struct MatMul<DenseMatrix<int64_t>, DenseMatrix<int64_t>, DenseMatrix<int64_t>> {
    static void apply(DenseMatrix<int64_t> *& res, const DenseMatrix<int64_t> * lhs, const DenseMatrix<int64_t> * rhs, DCTX(ctx)) {
        const auto nr1 = static_cast<int>(lhs->getNumRows());
        const auto nc1 = static_cast<int>(lhs->getNumCols());
        const auto nc2 = static_cast<int>(rhs->getNumCols());
        const auto nr2= static_cast<int>(rhs->getNumRows());
        assert((nc1 == static_cast<int>(rhs->getNumRows())) && "#cols of lhs and #rows of rhs must be the same");



        DenseMatrix<double> *m1=DataObjectFactory::create<DenseMatrix<double>>(nr1, nc1, false);   //lhs cast to double
        DenseMatrix<double> *m2=DataObjectFactory::create<DenseMatrix<double>>(nr2, nc2, false);   //rhs cast to double

        castObj<DenseMatrix<double>, DenseMatrix<int64_t>>(m1, lhs, nullptr);
        castObj<DenseMatrix<double>, DenseMatrix<int64_t>>(m2, rhs, nullptr);

       


        DenseMatrix<double> *doubleres= DataObjectFactory::create<DenseMatrix<double>>(nr1, nc2, false);

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<int64_t>>(nr1, nc2, false);

        if(nr1 == 1 && nc2 == 1) // Vector-Vector
            doubleres->set(0, 0, cblas_ddot(nc1, m1->getValues(), 1, m2->getValues(),
                static_cast<int>(m2->getRowSkip())));
        else if(nc2 == 1)        // Matrix-Vector
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nr1, nc1, 1, m1->getValues(),
            static_cast<int>(m1->getRowSkip()), m2->getValues(),static_cast<int>(m2->getRowSkip()), 0,
                doubleres->getValues(), static_cast<int>(doubleres->getRowSkip()));
        else                     // Matrix-Matrix
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nr1, nc2, nc1,
                    1, m1->getValues(), static_cast<int>(m1->getRowSkip()), m2->getValues(),
                    static_cast<int>(m2->getRowSkip()), 0, doubleres->getValues(), static_cast<int>(doubleres->getRowSkip()));

       castObj<DenseMatrix<int64_t>, DenseMatrix<double>>(res, doubleres, nullptr);
        
    }
};
