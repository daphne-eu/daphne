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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/MCSRMatrix.h>
#include <runtime/local/datastructures/CSCMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct EwUnaryMat {
    static void apply(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void ewUnaryMat(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    EwUnaryMat<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};


// ----------------------------------------------------------------------------
// MCSRMatrix <- MCSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<MCSRMatrix<VT>, MCSRMatrix<VT>> {
    static void apply(UnaryOpCode opCode, MCSRMatrix<VT> *& res, const MCSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t maxNumNonZeros = arg->getMaxNumNonZeros();

        if(res == nullptr)
            res = DataObjectFactory::create<MCSRMatrix<VT>>(numRows, numCols, maxNumNonZeros, true);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        for(size_t r = 0; r < numRows; r++) {
            const VT * rowValuesArg = arg->getValues(r);
            const size_t * colIdxsArg = arg->getColIdxs(r);
            size_t rowSize = arg->getNumNonZeros(r);

            for(size_t i = 0; i < rowSize; i++) {
                VT resultValue = func(rowValuesArg[i], ctx);
                if(resultValue != 0) { // Only store non-zero results
                    res->set(r, colIdxsArg[i], resultValue);
                }
            }
        }
    }
};




// ----------------------------------------------------------------------------
// CSCMatrix <- CSCMatrix
// ----------------------------------------------------------------------------


template<typename VT>
struct EwUnaryMat<CSCMatrix<VT>, CSCMatrix<VT>> {
    static void apply(UnaryOpCode opCode, CSCMatrix<VT> *& res, const CSCMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        // Ensure the result matrix is initialized
        if(res == nullptr)
            res = DataObjectFactory::create<CSCMatrix<VT>>(numRows, numCols, arg->getMaxNumNonZeros(), true);

        // Get function pointer for the unary operation
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        // Iterate over the columns of the CSCMatrix
        for(size_t c = 0; c < numCols; c++) {
            // Get non-zero values and their row indices for this column
            const VT* colValuesArg = arg->getValues(c);
            const size_t* rowIdxsArg = arg->getRowIdxs(c);
            size_t numNonZerosInCol = arg->getNumNonZeros(c);

            // Apply the unary operation on each non-zero value in the column
            for(size_t idx = 0; idx < numNonZerosInCol; idx++) {
                VT value = colValuesArg[idx];
                size_t rowIdx = rowIdxsArg[idx];

                // Compute the result and store it in the output matrix
                res->append(rowIdx, c, func(value, ctx));
            }
        }
    }
};



#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
