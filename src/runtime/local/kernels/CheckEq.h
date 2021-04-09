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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CHECKEQ_H
#define SRC_RUNTIME_LOCAL_KERNELS_CHECKEQ_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct CheckEq {
    static bool apply(const DT * lhs, const DT * rhs) = delete;
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
bool checkEq(const DT * lhs, const DT * rhs) {
    return CheckEq<DT>::apply(lhs, rhs);
};

// ****************************************************************************
// Operator == for matrices of the same type
// ****************************************************************************

template<class DT>
bool operator==(const DT & lhs, const DT & rhs) {
    return checkEq(&lhs, &rhs);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// Note that we do not use the generic `get` interface to matrices here since
// this operators is meant to be used for writing tests for, besides others,
// those generic interfaces.

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs) {
        if(lhs == rhs)
            return true;
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;
        
        const VT * valuesLhs = lhs->getValues();
        const VT * valuesRhs = rhs->getValues();
        
        const size_t rowSkipLhs = lhs->getRowSkip();
        const size_t rowSkipRhs = rhs->getRowSkip();
        
        if(valuesLhs == valuesRhs && rowSkipLhs == rowSkipRhs)
            return true;
        
        if(rowSkipLhs == numCols && rowSkipRhs == numCols)
            return !memcmp(valuesLhs, valuesRhs, numRows * numCols * sizeof(VT));
        else {
            for(size_t r = 0; r < numRows; r++) {
                if(memcmp(valuesLhs, valuesRhs, numCols * sizeof(VT)))
                    return false;
                valuesLhs += rowSkipLhs;
                valuesRhs += rowSkipRhs;
            }
            return true;
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEq<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs) {
        if(lhs == rhs)
            return true;
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;
        
        const VT * valuesBegLhs = lhs->getValues(0);
        const VT * valuesEndLhs = lhs->getValues(numRows);
        const VT * valuesBegRhs = rhs->getValues(0);
        const VT * valuesEndRhs = rhs->getValues(numRows);
        
        const size_t nnzLhs = valuesEndLhs - valuesBegLhs;
        const size_t nnzRhs = valuesEndRhs - valuesBegRhs;
        
        if(nnzLhs != nnzRhs)
            return false;
        
        if(valuesBegLhs != valuesBegRhs)
            if(memcmp(valuesBegLhs, valuesBegRhs, nnzLhs * sizeof(VT)))
                return false;
        
        const size_t * colIdxsBegLhs = lhs->getColIdxs(0);
        const size_t * colIdxsBegRhs = rhs->getColIdxs(0);
        
        if(colIdxsBegLhs != colIdxsBegRhs)
            if(memcmp(colIdxsBegLhs, colIdxsBegRhs, nnzLhs * sizeof(size_t)))
                return false;
        
        return true;
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_CHECKEQ_H