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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, typename VTRhs> struct EwBinaryObjSca {
    static void apply(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, VTRhs rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, typename VTRhs>
void ewBinaryObjSca(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, VTRhs rhs, DCTX(ctx)) {
    EwBinaryObjSca<DTRes, DTLhs, VTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTLhs, typename VTRhs>
struct EwBinaryObjSca<DenseMatrix<VTRes>, DenseMatrix<VTLhs>, VTRhs> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTRes> *&res, const DenseMatrix<VTLhs> *lhs, VTRhs rhs,
                      DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        const VTLhs *valuesLhs = lhs->getValues();
        VTRes *valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> func = getEwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs>(opCode);

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesLhs[c], rhs, ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, scalar
// ----------------------------------------------------------------------------
template <typename VTRes, typename VTLhs, typename VTRhs>
struct EwBinaryObjSca<CSRMatrix<VTRes>, CSRMatrix<VTLhs>, VTRhs> {
    static void apply(BinaryOpCode opCode, CSRMatrix<VTRes> *&res, const CSRMatrix<VTLhs> *lhs, VTRhs rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        const size_t nnzLhs = lhs->getNumNonZeros();

        EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> func = getEwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs>(opCode);

        const VTRes zeroRes = func(VTLhs(0), rhs, ctx);
        const bool isSparsityPreserving = zeroRes == VTRes(0);
        const size_t maxNnzRes = isSparsityPreserving ? nnzLhs : numRows * numCols;

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VTRes>>(numRows, numCols, maxNnzRes, false);

        VTRes *valuesRes = res->getValues();
        size_t *colIdxsRes = res->getColIdxs();
        size_t *rowOffsetsRes = res->getRowOffsets();

        const VTLhs *valuesLhs = lhs->getValues();
        const size_t *colIdxsLhs = lhs->getColIdxs();
        const size_t *rowOffsetsLhs = lhs->getRowOffsets();

        size_t nnzRes = 0;
        rowOffsetsRes[0] = 0;

        if (isSparsityPreserving) {
            for (size_t r = 0; r < numRows; ++r) {
                for (size_t i = rowOffsetsLhs[r]; i < rowOffsetsLhs[r + 1]; ++i) {
                    VTRes tmp = func(valuesLhs[i], rhs, ctx);
                    if (tmp != VTRes(0)) {
                        valuesRes[nnzRes] = tmp;
                        colIdxsRes[nnzRes] = colIdxsLhs[i];
                        ++nnzRes;
                    }
                }
                rowOffsetsRes[r + 1] = nnzRes;
            }
            return;
        }

        // Sparsity is not preserved.
        for (size_t r = 0; r < numRows; ++r) {
            size_t i = rowOffsetsLhs[r];
            for (size_t c = 0; c < numCols; ++c) {
                const bool hasVal = (i < rowOffsetsLhs[r + 1] && colIdxsLhs[i] == c);
                VTLhs lhsVal = hasVal ? valuesLhs[i] : VTLhs(0);
                if (hasVal)
                    ++i;
                VTRes tmp = func(lhsVal, rhs, ctx);
                if (tmp != VTRes(0)) {
                    valuesRes[nnzRes] = tmp;
                    colIdxsRes[nnzRes] = c;
                    ++nnzRes;
                }
            }
            rowOffsetsRes[r + 1] = nnzRes;
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTLhs, typename VTRhs>
struct EwBinaryObjSca<DenseMatrix<VTRes>, CSRMatrix<VTLhs>, VTRhs> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTRes> *&res, const CSRMatrix<VTLhs> *lhs, VTRhs rhs,
                      DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        VTRes *valuesRes = res->getValues();
        const size_t rowSkipRes = res->getRowSkip();

        EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> func = getEwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs>(opCode);

        const VTRes zeroRes = func(VTLhs(0), rhs, ctx);
        const bool isSparsityPreserving = (zeroRes == VTRes(0));

        if (isSparsityPreserving)
            memset(res->getValues(), 0, sizeof(VTRes) * numRows * numCols);
        else
            std::fill(res->getValues(), res->getValues() + numRows * numCols, zeroRes);

        // Process only non-zero positions
        for (size_t r = 0; r < numRows; r++) {
            const size_t nnzRow = lhs->getNumNonZeros(r);
            const VTLhs *valuesRowLhs = lhs->getValues(r);
            const size_t *colIdxsRowLhs = lhs->getColIdxs(r);

            for (size_t i = 0; i < nnzRow; i++) {
                const size_t c = colIdxsRowLhs[i];
                valuesRes[c] = func(valuesRowLhs[i], rhs, ctx);
            }

            valuesRes += rowSkipRes;
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar
// ----------------------------------------------------------------------------

template <typename VT> struct EwBinaryObjSca<Matrix<VT>, Matrix<VT>, VT> {
    static void apply(BinaryOpCode opCode, Matrix<VT> *&res, const Matrix<VT> *lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        // TODO Choose matrix implementation depending on expected number of
        // non-zeros.
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(lhs->get(r, c), rhs, ctx));
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, scalar
// ----------------------------------------------------------------------------

template <typename VT>
void ewBinaryFrameColSca(BinaryOpCode opCode, Frame *&res, const Frame *lhs, VT rhs, size_t c, DCTX(ctx)) {
    auto *col_res = res->getColumn<VT>(c);
    auto *col_lhs = lhs->getColumn<VT>(c);
    ewBinaryObjSca<DenseMatrix<VT>, DenseMatrix<VT>, VT>(opCode, col_res, col_lhs, rhs, ctx);
}

template <typename VT> struct EwBinaryObjSca<Frame, Frame, VT> {
    static void apply(BinaryOpCode opCode, Frame *&res, const Frame *lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<Frame>(numRows, numCols, lhs->getSchema(), lhs->getLabels(), false);

        for (size_t c = 0; c < numCols; c++) {
            switch (lhs->getColumnType(c)) {
            // For all value types:
            case ValueTypeCode::F64:
                ewBinaryFrameColSca<double>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::F32:
                ewBinaryFrameColSca<float>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI64:
                ewBinaryFrameColSca<int64_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI32:
                ewBinaryFrameColSca<int32_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI8:
                ewBinaryFrameColSca<int8_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI64:
                ewBinaryFrameColSca<uint64_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI32:
                ewBinaryFrameColSca<uint32_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI8:
                ewBinaryFrameColSca<uint8_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            default:
                throw std::runtime_error("EwBinaryObjSca::apply: unknown value type code");
            }
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
