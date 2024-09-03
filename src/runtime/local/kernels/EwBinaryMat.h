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

#include <algorithm>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
struct EwBinaryMat {
    static void apply(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void ewBinaryMat(BinaryOpCode opCode, DTRes *& res, const DTLhs * lhs, const DTRhs * rhs, DCTX(ctx)) {
    EwBinaryMat<DTRes, DTLhs, DTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VTres, typename VTlhs, typename VTrhs>
struct EwBinaryMat<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTres> *& res, const DenseMatrix<VTlhs> * lhs, const DenseMatrix<VTrhs> * rhs, DCTX(ctx)) {
        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, false);
        
        const VTlhs * valuesLhs = lhs->getValues();
        const VTrhs * valuesRhs = rhs->getValues();
        VTres * valuesRes = res->getValues();
        
        EwBinaryScaFuncPtr<VTres, VTlhs, VTrhs> func = getEwBinaryScaFuncPtr<VTres, VTlhs, VTrhs>(opCode);
        
        if(numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
            // matrix op matrix (same size)
            for(size_t r = 0; r < numRowsLhs; r++) {
                for(size_t c = 0; c < numColsLhs; c++)
                    valuesRes[c] = func(valuesLhs[c], valuesRhs[c], ctx);
                valuesLhs += lhs->getRowSkip();
                valuesRhs += rhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else if(numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
            // matrix op row-vector
            for(size_t r = 0; r < numRowsLhs; r++) {
                for(size_t c = 0; c < numColsLhs; c++)
                    valuesRes[c] = func(valuesLhs[c], valuesRhs[c], ctx);
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else if(numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
            // matrix op col-vector
            for(size_t r = 0; r < numRowsLhs; r++) {
                for(size_t c = 0; c < numColsLhs; c++)
                    valuesRes[c] = func(valuesLhs[c], valuesRhs[0], ctx);
                valuesLhs += lhs->getRowSkip();
                valuesRhs += rhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        }
        else {
            throw std::runtime_error("EwBinaryMat(Dense) - lhs and rhs must either "
                "have the same dimensions, or one of them must be a row/column vector "
                "with the width/height of the other, but lhs has shape (" +
                std::to_string(numRowsLhs) + " x " + std::to_string(numColsLhs) +
                ") and rhs has shape (" + std::to_string(numRowsRhs) + " x " +
                std::to_string(numColsRhs) + ")");
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<DenseMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("EwBinaryMat(DenseMatrix <- CSRMatrix, CSRMatrix) - lhs and rhs must have the same dimensions.");
        }

        if (res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        }

        VT * valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        std::fill(valuesRes, valuesRes + numRows * numCols, VT(0));

        switch(opCode) {
            case BinaryOpCode::ADD: {
                for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                    size_t nnzRowRhs = rhs->getNumNonZeros(rowIdx);

                    const VT* valuesRowLhs = lhs->getValues(rowIdx);
                    const size_t* colIdxsRowLhs = lhs->getColIdxs(rowIdx);

                    const VT* valuesRowRhs = rhs->getValues(rowIdx);
                    const size_t* colIdxsRowRhs = rhs->getColIdxs(rowIdx);

                    size_t posLhs = 0, posRhs = 0;

                    while (posLhs < nnzRowLhs || posRhs < nnzRowRhs) {
                        if (posLhs < nnzRowLhs && (posRhs >= nnzRowRhs || colIdxsRowLhs[posLhs] < colIdxsRowRhs[posRhs])) {
                            // Only lhs has a value in this column
                            valuesRes[rowIdx * numCols + colIdxsRowLhs[posLhs]] = func(valuesRowLhs[posLhs], VT(0), ctx);
                            posLhs++;
                        }
                        else if (posRhs < nnzRowRhs && (posLhs >= nnzRowLhs || colIdxsRowRhs[posRhs] < colIdxsRowLhs[posLhs])) {
                            // Only rhs has a value in this column
                            valuesRes[rowIdx * numCols + colIdxsRowRhs[posRhs]] = func(VT(0), valuesRowRhs[posRhs], ctx);
                            posRhs++;
                        }
                        else {
                            // Both lhs and rhs have values in this column
                            valuesRes[rowIdx * numCols + colIdxsRowLhs[posLhs]] = func(valuesRowLhs[posLhs], valuesRowRhs[posRhs], ctx);
                            posLhs++;
                            posRhs++;
                        }
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("EwBinaryMat(DenseMatrix <- CSRMatrix, CSRMatrix) - unsupported BinaryOpCode");
        }
    }
};


// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        
        if((numRows != rhs->getNumRows() && rhs->getNumRows() != 1) || (numCols != rhs->getNumCols() && rhs->getNumCols() != 1))
            throw std::runtime_error("EwBinaryMat(Dense) - lhs and rhs must have the same dimensions (or broadcast)");
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        VT * valuesRes = res->getValues();
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);
        
        switch(opCode) {
            case BinaryOpCode::ADD: { // Add operation
                for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    auto rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
                    for(size_t colIdx = 0; colIdx < numCols; colIdx++) {
                        auto lhsVal = lhs->get(rowIdx, colIdx);
                        auto rhsVal = rhs->get(rhsRow, colIdx);
                        valuesRes[rowIdx * numCols + colIdx] = func(lhsVal, rhsVal, ctx);
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("EwBinaryMat(Dense) - unsupported BinaryOpCode");
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<DenseMatrix<VT>, DenseMatrix<VT>, CSRMatrix<VT>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if((numRows != rhs->getNumRows() && rhs->getNumRows() != 1) || (numCols != rhs->getNumCols() && rhs->getNumCols() != 1))
            throw std::runtime_error("EwBinaryMat(Dense) - lhs and rhs must have the same dimensions (or broadcast)");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        VT * valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        switch(opCode) {
            case BinaryOpCode::ADD: { // Add operation
                for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    auto lhsRow = (lhs->getNumRows() == 1 ? 0 : rowIdx);
                    for(size_t colIdx = 0; colIdx < numCols; colIdx++) {
                        auto lhsVal = lhs->get(lhsRow, colIdx);
                        auto rhsVal = rhs->get(rowIdx, colIdx);
                        valuesRes[rowIdx * numCols + colIdx] = func(lhsVal, rhsVal, ctx);
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("EwBinaryMat(Dense) - unsupported BinaryOpCode");
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<CSRMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(BinaryOpCode opCode, CSRMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const CSRMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        if( numRows != rhs->getNumRows() || numCols != rhs->getNumCols() )
            throw std::runtime_error("EwBinaryMat(CSR) - lhs and rhs must have the same dimensions.");
        
        size_t maxNnz;
        switch(opCode) {
            case BinaryOpCode::ADD: // merge
                maxNnz = lhs->getNumNonZeros() + rhs->getNumNonZeros();
                break;
            case BinaryOpCode::MUL: // intersect
                maxNnz = std::min(lhs->getNumNonZeros(), rhs->getNumNonZeros());
                break;
            default:
                throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }
        
        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, maxNnz, false);
        
        size_t * rowOffsetsRes = res->getRowOffsets();
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);
        
        rowOffsetsRes[0] = 0;
        
        switch(opCode) {
            case BinaryOpCode::ADD: { // merge non-zero cells
                for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                    size_t nnzRowRhs = rhs->getNumNonZeros(rowIdx);
                    if(nnzRowLhs && nnzRowRhs) {
                        // merge within row
                        const VT * valuesRowLhs = lhs->getValues(rowIdx);
                        const VT * valuesRowRhs = rhs->getValues(rowIdx);
                        VT * valuesRowRes = res->getValues(rowIdx);
                        const size_t * colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                        const size_t * colIdxsRowRhs = rhs->getColIdxs(rowIdx);
                        size_t * colIdxsRowRes = res->getColIdxs(rowIdx);
                        size_t posLhs = 0;
                        size_t posRhs = 0;
                        size_t posRes = 0;
                        while(posLhs < nnzRowLhs && posRhs < nnzRowRhs) {
                            if(colIdxsRowLhs[posLhs] == colIdxsRowRhs[posRhs]) {
                                valuesRowRes[posRes] = func(valuesRowLhs[posLhs], valuesRowRhs[posRhs], ctx);
                                colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                                posLhs++;
                                posRhs++;
                            }
                            else if(colIdxsRowLhs[posLhs] < colIdxsRowRhs[posRhs]) {
                                valuesRowRes[posRes] = valuesRowLhs[posLhs];
                                colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                                posLhs++;
                            }
                            else {
                                valuesRowRes[posRes] = valuesRowRhs[posRhs];
                                colIdxsRowRes[posRes] = colIdxsRowRhs[posRhs];
                                posRhs++;
                            }
                            posRes++;
                        }
                        // copy from left
                        const size_t restRowLhs = nnzRowLhs - posLhs;
                        memcpy(valuesRowRes + posRes, valuesRowLhs + posLhs, restRowLhs * sizeof(VT));
                        memcpy(colIdxsRowRes + posRes, colIdxsRowLhs + posLhs, restRowLhs * sizeof(size_t));
                        // copy from right
                        const size_t restRowRhs = nnzRowRhs - posRhs;
                        memcpy(valuesRowRes + posRes, valuesRowRhs + posRhs, restRowRhs * sizeof(VT));
                        memcpy(colIdxsRowRes + posRes, colIdxsRowRhs + posRhs, restRowRhs * sizeof(size_t));
                        
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes + restRowLhs + restRowRhs;
                    }
                    else if(nnzRowLhs) {
                        // copy from left
                        memcpy(res->getValues(rowIdx), lhs->getValues(rowIdx), nnzRowLhs * sizeof(VT));
                        memcpy(res->getColIdxs(rowIdx), lhs->getColIdxs(rowIdx), nnzRowLhs * sizeof(size_t));
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + nnzRowLhs;
                    }
                    else if(nnzRowRhs) {
                        // copy from right
                        memcpy(res->getValues(rowIdx), rhs->getValues(rowIdx), nnzRowRhs * sizeof(VT));
                        memcpy(res->getColIdxs(rowIdx), rhs->getColIdxs(rowIdx), nnzRowRhs * sizeof(size_t));
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + nnzRowRhs;
                    }
                    else
                        // empty row in result
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
                }
                break;
            }
            case BinaryOpCode::MUL: { // intersect non-zero cells
                for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                    size_t nnzRowRhs = rhs->getNumNonZeros(rowIdx);
                    if(nnzRowLhs && nnzRowRhs) {
                        // intersect within row
                        const VT * valuesRowLhs = lhs->getValues(rowIdx);
                        const VT * valuesRowRhs = rhs->getValues(rowIdx);
                        VT * valuesRowRes = res->getValues(rowIdx);
                        const size_t * colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                        const size_t * colIdxsRowRhs = rhs->getColIdxs(rowIdx);
                        size_t * colIdxsRowRes = res->getColIdxs(rowIdx);
                        size_t posLhs = 0;
                        size_t posRhs = 0;
                        size_t posRes = 0;
                        while(posLhs < nnzRowLhs && posRhs < nnzRowRhs) {
                            if(colIdxsRowLhs[posLhs] == colIdxsRowRhs[posRhs]) {
                                valuesRowRes[posRes] = func(valuesRowLhs[posLhs], valuesRowRhs[posRhs], ctx);
                                colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                                posLhs++;
                                posRhs++;
                                posRes++;
                            }
                            else if(colIdxsRowLhs[posLhs] < colIdxsRowRhs[posRhs])
                                posLhs++;
                            else
                                posRhs++;
                        }
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
                    }
                    else
                        // empty row in result
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
                }
                break;
            }
            default:
                throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }
        
        // TODO Update number of non-zeros in result in the end.
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<CSRMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>> {
    static void apply(BinaryOpCode opCode, CSRMatrix<VT> *& res, const CSRMatrix<VT> * lhs, const DenseMatrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        // TODO: lhs broadcast
        if( (numRows != rhs->getNumRows() &&  rhs->getNumRows() != 1)
            || (numCols != rhs->getNumCols() && rhs->getNumCols() != 1 ) )
            throw std::runtime_error("EwBinaryMat(CSR) - lhs and rhs must have the same dimensions (or broadcast)");

        size_t maxNnz;
        switch(opCode) {
        case BinaryOpCode::MUL: // intersect
            maxNnz = lhs->getNumNonZeros();
            break;
        case BinaryOpCode::ADD:
            maxNnz = lhs->getNumNonZeros() + rhs->getNumNonZeros();
            break;
        default:
            throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }

        if(res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, maxNnz, false);

        size_t *rowOffsetsRes = res->getRowOffsets();
        rowOffsetsRes[0] = 0;

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        switch(opCode) {
            case BinaryOpCode::MUL: { // intersect non-zero cells
                for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                    if(nnzRowLhs) {
                        // intersect within row
                        const VT * valuesRowLhs = lhs->getValues(rowIdx);
                        VT * valuesRowRes = res->getValues(rowIdx);
                        const size_t * colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                        size_t * colIdxsRowRes = res->getColIdxs(rowIdx);
                        auto rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
                        size_t posRes = 0;
                        for (size_t posLhs = 0; posLhs < nnzRowLhs; ++posLhs) {
                            auto rhsCol = (rhs->getNumCols() == 1 ? 0 : colIdxsRowLhs[posLhs]);
                            auto rVal = rhs->get(rhsRow, rhsCol);
                            if(rVal != 0) {
                                valuesRowRes[posRes] = func(valuesRowLhs[posLhs], rVal, ctx);
                                colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                                posRes++;
                            }
                        }
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
                    }
                    else
                        // empty row in result
                        rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
                }
                break;
            }
            case BinaryOpCode::ADD: {
                VT* valuesRes = res->getValues(0);  // Pointer to the result values array
                size_t* colIdxsRes = res->getColIdxs(0);  // Pointer to the result column indices array
                size_t posRes = 0;  // Track position in the result matrix's values and colIdxs arrays

                for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    const size_t rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
                    size_t posLhs = 0;  // Current position in LHS row
                    size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                    const VT* valuesRowLhs = lhs->getValues(rowIdx);
                    const size_t* colIdxsRowLhs = lhs->getColIdxs(rowIdx);

                    size_t colIdxRhs = 0;  // Start from the first column index for the dense matrix

                    // Merge LHS and RHS
                    while (colIdxRhs < numCols) {
                        VT lhsVal = 0;
                        VT rhsVal = rhs->get(rhsRow, colIdxRhs);

                        // Check if there's a corresponding LHS value
                        if (posLhs < nnzRowLhs && colIdxsRowLhs[posLhs] == colIdxRhs) {
                            lhsVal = valuesRowLhs[posLhs++];
                        }

                        VT resultVal = func(lhsVal, rhsVal, ctx);

                        // Only store non-zero results
                        if (resultVal != 0) {
                            valuesRes[posRes] = resultVal;
                            colIdxsRes[posRes] = colIdxRhs;
                            posRes++;
                        }
                        colIdxRhs++;
                    }
                    rowOffsetsRes[rowIdx + 1] = posRes;
                }
                break;
            }
           
            default:
                throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
            }

        // TODO Update number of non-zeros in result in the end.
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwBinaryMat<Matrix<VT>, Matrix<VT>, Matrix<VT>> {
    static void apply(BinaryOpCode opCode, Matrix<VT> *& res, const Matrix<VT> * lhs, const Matrix<VT> * rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            throw std::runtime_error("EwBinaryMat - lhs and rhs must have the same dimensions.");
        
        // TODO Choose matrix implementation depending on expected number of non-zeros.
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);
        
        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(lhs->get(r, c), rhs->get(r, c), ctx));
        res->finishAppend();
    }
};
