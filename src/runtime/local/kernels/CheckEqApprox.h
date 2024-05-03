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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

// TODO This kernel should handle integral value types gracefully, e.g. by
// forwarding to the non-approximate checkEq-kernel. This would allow us to
// easily use the checkEqApprox-kernel in test cases without differentiating
// integral and floating-point value types.

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct CheckEqApprox{
    static bool apply(const DT * lhs, const DT * rhs, double eps, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks if the two given matrices are approximately equal.
 * 
 * More precisely, this requires that they have the same dimensions and are approximately
 * elementwise equal, i.e., if the difference between two elements is not greater than the threshold `eps`,
 * they are considered as equal. 
 *
 * @param lhs The first matrix.
 * @param rhs The second matrix.
 * @param eps The similarity threshold.
 * @return `true` if they are equal, `false` otherwise.
 */
template<class DT>
bool checkEqApprox(const DT * lhs, const DT * rhs, double eps, DCTX(ctx)) {
    return CheckEqApprox<DT>::apply(lhs, rhs, eps, ctx);
}

/*
// ****************************************************************************
// Operator == for matrices of the same type
// ****************************************************************************

template<class DT>
bool operator==(const DT & lhs, const DT & rhs) {
     double eps= 0.000001; //required for the equal operator
    // nullptr might become problematic some day.
    return checkEqApprox(&lhs, &rhs, eps, nullptr);
}
*/

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// Note that we do not use the generic `get` interface to matrices here since
// this operator is meant to be used for writing tests for, besides others,
// those generic interfaces.

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEqApprox<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, double eps, DCTX(ctx)) {
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
                if (diff> eps)
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
    static bool apply(const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, double eps, DCTX(ctx)) {
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
                if (diff> eps)
                    return false;
            }   
        }
        return true;

    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct CheckEqApprox<Frame> {
    static bool apply(const Frame * lhs, const Frame * rhs, double eps, DCTX(ctx)) {
        if(lhs == rhs)
            return true;
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;
        
        if(memcmp(lhs->getSchema(), rhs->getSchema(), numCols * sizeof(ValueTypeCode)) != 0)
            return false;

        const std::string * labelsLhs = lhs->getLabels();
        const std::string * labelsRhs = rhs->getLabels();
        for (size_t c = 0; c < numCols; c++) {
            if(labelsLhs[c] != labelsRhs[c])
                return false;
        }
        
        for (size_t c = 0; c < numCols; c++)
        {
            switch(lhs->getColumnType(c)) {
                // For all value types:
                case ValueTypeCode::F64: if(!checkEqApprox(lhs->getColumn<double>(c),
                    rhs->getColumn<double>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::F32: if (!checkEqApprox(lhs->getColumn<float>(c),
                    rhs->getColumn<float>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::SI64: if (!checkEqApprox(lhs->getColumn<int64_t>(c),
                    rhs->getColumn<int64_t>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::SI32: if (!checkEqApprox(lhs->getColumn<int32_t>(c),
                    rhs->getColumn<int32_t>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::SI8 : if (!checkEqApprox(lhs->getColumn<int8_t>(c),
                    rhs->getColumn<int8_t>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::UI64: if (!checkEqApprox(lhs->getColumn<uint64_t>(c),
                    rhs->getColumn<uint64_t>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::UI32: if (!checkEqApprox(lhs->getColumn<uint32_t>(c), 
                    rhs->getColumn<uint32_t>(c), eps, ctx)) return false;
                    break;
                case ValueTypeCode::UI8 : if (!checkEqApprox(lhs->getColumn<uint8_t>(c),
                    rhs->getColumn<uint8_t>(c), eps, ctx)) return false;
                    break;
                default:
                    throw std::runtime_error("CheckEqApprox::apply: unknown value type code");
            }
        }   
        return true;
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct CheckEqApprox<Matrix<VT>> {
    static bool apply(const Matrix<VT> * lhs, const Matrix<VT> * rhs, double eps, DCTX(ctx)) {
        if (lhs == rhs)
            return true;
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            return false;

        for (size_t r=0; r < numRows; ++r) {
            for (size_t c=0; c < numCols; ++c) {
                double diff = lhs->get(r, c) - rhs->get(r, c);
                if (std::abs(diff) > eps)
                    return false;
            }
        }        
         
        return true;
    }
};