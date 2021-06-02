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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H
#define SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H

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
    static void apply(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void matMul(DTRes *& res, const DTLhs * lhs, const DTRhs * rhs) {
    MatMul<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct MatMul<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, const DenseMatrix<float> * lhs, const DenseMatrix<float> * rhs) {
        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        assert((numColsLhs == numRowsRhs) && "#cols of lhs and #rows of rhs must be the same");
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<float>>(numRowsLhs, numColsRhs, false);
        
        cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                numRowsLhs, numColsRhs, numColsLhs,
                1, lhs->getValues(), lhs->getRowSkip(),
                rhs->getValues(), rhs->getRowSkip(),
                0, res->getValues(), res->getRowSkip()
        );
    }
};

template<>
struct MatMul<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, const DenseMatrix<double> * lhs, const DenseMatrix<double> * rhs) {
        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        assert((numColsLhs == numRowsRhs) && "#cols of lhs and #rows of rhs must be the same");
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<double>>(numRowsLhs, numColsRhs, false);
        
        cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                numRowsLhs, numColsRhs, numColsLhs,
                1, lhs->getValues(), lhs->getRowSkip(),
                rhs->getValues(), rhs->getRowSkip(),
                0, res->getValues(), res->getRowSkip()
        );
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_MATMUL_H