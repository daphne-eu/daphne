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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>

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
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------
template<typename VT>
struct EwUnaryMat<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(UnaryOpCode opCode, CSRMatrix<VT> *& res, const CSRMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t numNonZeros = arg->getNumNonZeros();

        // Ensure res is initialized with enough space
        if (res == nullptr) {
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros * 1.5, false);
        }

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        size_t* rowOffsetsRes = res->getRowOffsets();
        rowOffsetsRes[0] = 0;

        size_t posRes = 0;  // Track the position in the result matrix's values and colIdxs arrays

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            size_t nnzRowArg = arg->getNumNonZeros(rowIdx);
            const VT* valuesRowArg = arg->getValues(rowIdx);
            const size_t* colIdxsRowArg = arg->getColIdxs(rowIdx);

            for (size_t posArg = 0; posArg < nnzRowArg; ++posArg) {
                VT value = func(valuesRowArg[posArg], ctx);

                // Store non-zero values in result matrix
                if (value != VT(0)) {
                    if (posRes >= res->getMaxNumNonZeros()) {
                        // Resize the result matrix to accommodate more non-zero values
                        res->resize(posRes + nnzRowArg);
                    }
                    res->getValues()[posRes] = value;
                    res->getColIdxs()[posRes] = colIdxsRowArg[posArg];
                    posRes++;
                }
            }

            // Handle zero values that become non-zero
            for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                if (std::find(colIdxsRowArg, colIdxsRowArg + nnzRowArg, colIdx) == colIdxsRowArg + nnzRowArg) {
                    // colIdx not found in the current non-zero column indices
                    VT value = func(VT(0), ctx);  // Apply function to a zero value
                    if (value != VT(0)) {
                        if (posRes >= res->getMaxNumNonZeros()) {
                            // Resize the result matrix to accommodate more non-zero values
                            res->resize(posRes + 1);
                        }
                        res->getValues()[posRes] = value;
                        res->getColIdxs()[posRes] = colIdx;
                        posRes++;
                    }
                }
            }

            rowOffsetsRes[rowIdx + 1] = posRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<Matrix<VT>, Matrix<VT>> {
    static void apply(UnaryOpCode opCode, Matrix<VT> *& res, const Matrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(arg->get(r, c), ctx));
        res->finishAppend();
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
