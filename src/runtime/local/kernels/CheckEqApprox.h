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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CHECKEQAPPROX_H
#define SRC_RUNTIME_LOCAL_KERNELS_CHECKEQAPPROX_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>


#include <cstddef>
#include <cstring>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct CheckEqApprox{
    static bool apply(const DT * lhs, const DT * rhs, double esp, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks if the two given matrices are Approximately equal.
 * 
 * More precisely, this requires that they have the same dimensions and are approximately
 * elementwise equal, i.e., if the difference between two elements is less than the threshold EPS,
 * they are considered as equal. 
 *
 * @param lhs The first matrix.
 * @param rhs The second matrix.
 * @return `true` if they are equal, `false` otherwise.
 */
template<class DT>
bool checkEqApprox(const DT * lhs, const DT * rhs, double esp, DCTX(ctx)) {
    return CheckEqApprox<DT>::apply(lhs, rhs, esp, ctx);
};

// ****************************************************************************
// Operator == for matrices of the same type
// ****************************************************************************

template<class DT>
bool operator==(const DT & lhs, const DT & rhs) {
     double eps= 0.000001; //required for the equal operator
    // nullptr might become problematic some day.
    return checkEqApprox(&lhs, &rhs, eps, nullptr);
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
struct CheckEqApprox<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, double esp, DCTX(ctx)) {
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
        
        if(valuesLhs == valuesRhs && rowSkipLhs == rowSkipRhs) // same pointer
            return true;
        
        for(size_t r = 0; r < numRows; r++){
            for(size_t c = 0; c < numCols; c++){
                VT diff = valuesLhs[c] - valuesRhs[c];
                if (diff==0)
                    continue;
                diff = diff>0? diff : -diff;
                if (diff> esp)
                    return false;
            }   
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
        }        
         
       return true;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEqApprox<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, double esp, DCTX(ctx)) {
        if(lhs == rhs)
            return true;
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;
       
        for(size_t r = 0; r < numRows; r++){
            const VT * valuesLhs = lhs->getValues(r);
            const VT * valuesRhs = rhs->getValues(r);
            const size_t nnzElementsLhs= lhs->getNumNonZeros(r);
            const size_t nnzElementsRhs= rhs->getNumNonZeros(r);
            if (nnzElementsLhs!=nnzElementsRhs)
                return false;
            for(size_t c = 0; c < nnzElementsLhs; c++){
                VT diff = valuesLhs[c] - valuesRhs[c];
                if (diff==0)
                     continue;
                diff = diff>0? diff : -diff;
                if (diff> esp) 
                    return false;
            }   
        }
        return true;

    }
};
#endif //SRC_RUNTIME_LOCAL_KERNELS_CHECKEQAPPROX_H
