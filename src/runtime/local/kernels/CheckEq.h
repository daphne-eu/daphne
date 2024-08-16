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
#include <runtime/local/datastructures/COOMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/Matrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct CheckEq {
    static bool apply(const DT * lhs, const DT * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks if the two given matrices are logically equal.
 * 
 * More precisely, this requires that they have the same dimensions and are
 * elementwise equal.
 * 
 * @param lhs The first matrix.
 * @param rhs The second matrix.
 * @return `true` if they are equal, `false` otherwise.
 */
template<class DT>
bool checkEq(const DT * lhs, const DT * rhs, DCTX(ctx)) {
    return CheckEq<DT>::apply(lhs, rhs, ctx);
};

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};

// ----------------------------------------------------------------------------
// COOMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<COOMatrix<VT>> {
    static bool apply(const COOMatrix<VT> * lhs, const COOMatrix<VT> * rhs, DCTX(ctx)) {
        if(lhs == rhs)
            return true;

        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;

        const VT * valuesLhs = lhs->getValues();
        const size_t * rowsLhs = lhs->getRowIdxs();
        const size_t * colsLhs = lhs->getColIdxs();

        const VT * valuesRhs = rhs->getValues();
        const size_t * rowsRhs = rhs->getRowIdxs();
        const size_t * colsRhs = rhs->getColIdxs();

        const size_t nnzLhs = lhs->getNumNonZeros();
        const size_t nnzRhs = rhs->getNumNonZeros();

        size_t lowerRowLhs = lhs->getLowerRow();
        size_t lowerRowRhs = rhs->getLowerRow();

        if(nnzLhs != nnzRhs)
            return false;

        for (size_t i = 0; i < nnzLhs; i++) {
            const size_t rowLhs = rowsLhs[i] - lowerRowLhs;
            const size_t rowRhs = rowsRhs[i] - lowerRowRhs;
            if (rowLhs != rowRhs) return false;
            const size_t colLhs = colsLhs[i];
            const size_t colRhs = colsRhs[i];
            if (colLhs != colRhs) return false;
            const VT valLhs = valuesLhs[i];
            const VT valRhs = valuesRhs[i];
            if (valLhs != valRhs) return false;
        }
        return true;
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct CheckEq<Frame> {
    static bool apply(const Frame * lhs, const Frame * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};

// ----------------------------------------------------------------------------
// Contiguous Tensor
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<ContiguousTensor<VT>> {
    static bool apply(const ContiguousTensor<VT> * lhs, const ContiguousTensor<VT> * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};

// ----------------------------------------------------------------------------
// Chunked Tensor
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<ChunkedTensor<VT>> {
    static bool apply(const ChunkedTensor<VT> * lhs, const ChunkedTensor<VT> * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<Matrix<VT>> {
    static bool apply(const Matrix<VT> * lhs, const Matrix<VT> * rhs, DCTX(ctx)) {
        return *lhs == *rhs;
    }
};