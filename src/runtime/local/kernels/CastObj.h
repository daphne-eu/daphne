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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/CastSca.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct CastObj {
    static void apply(DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Performs a cast of the given data object to another type.
 *
 * @param arg The data object to cast.
 * @return The casted data object.
 */
template <class DTRes, class DTArg> void castObj(DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    CastObj<DTRes, DTArg>::apply(res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- Frame
// ----------------------------------------------------------------------------

template <typename VTRes> class CastObj<DenseMatrix<VTRes>, Frame> {

    /**
     * @brief Casts the values of the input column at index `c` and stores the
     * casted values to column `c` in the output matrix.
     * @param res The output matrix.
     * @param argFrm The input frame.
     * @param c The position of the column to cast.
     */
    template <typename VTArg> static void castCol(DenseMatrix<VTRes> *res, const Frame *argFrm, size_t c) {
        const size_t numRows = argFrm->getNumRows();
        const DenseMatrix<VTArg> *argCol = argFrm->getColumn<VTArg>(c);
        for (size_t r = 0; r < numRows; r++)
            res->set(r, c, castSca<VTRes, VTArg>(argCol->get(r, 0), nullptr));
        DataObjectFactory::destroy(argCol);
    }

  public:
    static void apply(DenseMatrix<VTRes> *&res, const Frame *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numCols == 1 && arg->getColumnType(0) == ValueTypeUtils::codeFor<VTRes>) {
            // The input frame has a single column of the result's value type.
            // Zero-cost cast from frame to matrix.
            // TODO This case could even be used for (un)signed integers of the
            // same width, involving a reinterpret cast of the pointers.
            // TODO Can we avoid this const_cast?
            res = const_cast<DenseMatrix<VTRes> *>(arg->getColumn<VTRes>(0));
        } else {
            // The input frame has multiple columns and/or other value types
            // than the result.
            // Need to change column-major to row-major layout and/or cast the
            // individual values.
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);
            // TODO We could run over the rows in blocks for cache efficiency.
            for (size_t c = 0; c < numCols; c++) {
                // TODO We do not really need all cases.
                // - All pairs of the same type can be handled by a single
                //   copy-the-column helper function.
                // - All pairs of (un)signed integer types of the same width
                //   as well.
                // - Truncating integers to a narrower type does not need to
                //   consider (un)signedness either.
                // - ...
                switch (arg->getColumnType(c)) {
                // For all value types:
                case ValueTypeCode::F64:
                    castCol<double>(res, arg, c);
                    break;
                case ValueTypeCode::F32:
                    castCol<float>(res, arg, c);
                    break;
                case ValueTypeCode::SI64:
                    castCol<int64_t>(res, arg, c);
                    break;
                case ValueTypeCode::SI32:
                    castCol<int32_t>(res, arg, c);
                    break;
                case ValueTypeCode::SI8:
                    castCol<int8_t>(res, arg, c);
                    break;
                case ValueTypeCode::UI64:
                    castCol<uint64_t>(res, arg, c);
                    break;
                case ValueTypeCode::UI32:
                    castCol<uint32_t>(res, arg, c);
                    break;
                case ValueTypeCode::UI8:
                    castCol<uint8_t>(res, arg, c);
                    break;
                case ValueTypeCode::STR:
                    castCol<std::string>(res, arg, c);
                    break;
                default:
                    throw std::runtime_error("CastObj::apply: unknown value type code");
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
//  Frame <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTArg> class CastObj<Frame, DenseMatrix<VTArg>> {

  public:
    static void apply(Frame *&res, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        std::vector<Structure *> cols;
        if (numCols == 1 && arg->getRowSkip() == 1) {
            // The input matrix has a single column and is not a view into a
            // column range of another matrix, so it can be reused as the
            // column matrix of the output frame.
            // Cheap/Low-cost cast from dense matrix to frame.
            cols.push_back(const_cast<DenseMatrix<VTArg> *>(arg));
        } else {
            // The input matrix has multiple columns.
            // Need to change row-major to column-major layout and
            // split matrix into single column matrices.
            for (size_t c = 0; c < numCols; c++) {
                auto *colMatrix = DataObjectFactory::create<DenseMatrix<VTArg>>(numRows, 1, false);
                for (size_t r = 0; r < numRows; r++)
                    colMatrix->set(r, 0, arg->get(r, c));
                cols.push_back(colMatrix);
            }
        }
        res = DataObjectFactory::create<Frame>(cols, nullptr);
    }
};

// ----------------------------------------------------------------------------
//  DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> class CastObj<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

  public:
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        auto resVals = res->getValues();
        auto argVals = arg->getValues();

        if (arg->getRowSkip() == numCols && res->getRowSkip() == numCols)
            // Since DenseMatrix implementation is backed by
            // a single dense array of values, we can simply
            // perform cast in one loop over that array.
            for (size_t idx = 0; idx < numCols * numRows; idx++)
                resVals[idx] = castSca<VTRes, VTArg>(argVals[idx], ctx);
        else
            // res and arg might be views into a larger DenseMatrix.
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    resVals[c] = castSca<VTRes, VTArg>(argVals[c], ctx);
                resVals += res->getRowSkip();
                argVals += arg->getRowSkip();
            }
    }
};

// ----------------------------------------------------------------------------
//  DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<DenseMatrix<VT>, CSRMatrix<VT>> {

  public:
    static void apply(DenseMatrix<VT> *&res, const CSRMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        const size_t numNonZeros = arg->getNumNonZeros();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        VT *valuesRes = res->getValues();

        if (numNonZeros == 0)
            // Special case: The arg has no non-zeros.
            // We can simply fill the entire res with zeros, no need to read even the arg's rowOffsets.
            std::fill(valuesRes, valuesRes + numRows * numCols, 0);
        else if (numNonZeros == numRows * numCols) {
            // Special case: The arg has no zeros.
            // We can simply use the arg's values as-is for the res.
            // TODO We could even share the arg's values array with the res to avoid copying.
            const VT *valuesArg = arg->getValues(0);
            std::copy(valuesArg, valuesArg + numNonZeros, valuesRes);
        } else {
            // General case: The arg has zeros and non-zeros.
            // We transfer the arg's non-zeros to the res and fill the gaps with zeros.
            size_t prevPosRes = 0;
            size_t posResRow = 0;
            for (size_t r = 0; r < numRows; r++) {
                const size_t numNonZerosRow = arg->getNumNonZeros(r);
                const VT *valuesArgRow = arg->getValues(r);
                const size_t *colIdxsArgRow = arg->getColIdxs(r);
                for (size_t i = 0; i < numNonZerosRow; i++) {
                    const size_t posRes = posResRow + colIdxsArgRow[i];
                    std::fill(valuesRes + prevPosRes, valuesRes + posRes, 0);
                    valuesRes[posRes] = valuesArgRow[i];
                    prevPosRes = posRes + 1;
                }
                posResRow += numCols;
            }
            std::fill(valuesRes + prevPosRes, valuesRes + numRows * numCols, 0);
        }
    }
};

// ----------------------------------------------------------------------------
//  CSRMatrix  <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<CSRMatrix<VT>, DenseMatrix<VT>> {

  public:
    static void apply(CSRMatrix<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();
        size_t numNonZeros = 0;

        // Step 1: Count the non-zeros in the argument DenseMatrix.
        // Required to know how much memory to allocate for the result CSRMatrix.
        const VT *valuesArg = arg->getValues();
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                if (valuesArg[c] != 0)
                    numNonZeros++;
            valuesArg += arg->getRowSkip();
        }

        // Step 2: Create and populate the result CSRMatrix.
        if (numNonZeros == 0) {
            // Special case: There are no non-zeros in the argument.
            // This case is handled more efficiently than the common case.

            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, false);

            // The result's values and colIdx arrays are empty.

            // The result's rowOffsets array must be filled with zeros.
            size_t *rowOffsetsRes = res->getRowOffsets();
            std::fill(rowOffsetsRes, rowOffsetsRes + numRows + 1, 0);
        } else if (numNonZeros == numRows * numCols) {
            // Special case: There are no zeros in the argument.
            // This case is handled more efficiently than the common case.

            // Create the result CSRMatrix and ensure that the values array is populated.
            if (arg->getRowSkip() == numCols || numRows == 1) {
                // The data in arg is contiguous in memory. The result can share the data with the argument.
                if (res == nullptr)
                    res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros,
                                                                   arg->getValuesSharedPtr());
                // TODO Should we ever pass an existing data object for the result, we must make sure that either (a)
                // the values array of the argument is copied, or (b) the values array of the result is destroyed and
                // replaced by the argument's values array.
            } else {
                // The data in arg is not contiguous in memory. We must copy the data row by row from the argument to
                // the result.
                if (res == nullptr)
                    res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, false);
                const VT *valuesArg = arg->getValues();
                VT *valuesRes = res->getValues();
                for (size_t r = 0; r < numRows; r++) {
                    std::copy(valuesArg, valuesArg + numCols, valuesRes);
                    valuesArg += arg->getRowSkip();
                    valuesRes += numCols;
                }
            }

            // The result's colIdxs are ascending sequences per row; the rowOffsets are the multiples of the number of
            // columns.
            size_t *colIdxsRes = res->getColIdxs();
            size_t *rowOffsetsRes = res->getRowOffsets();
            for (size_t r = 0; r < numRows; r++) {
                std::iota(colIdxsRes, colIdxsRes + numCols, 0);
                colIdxsRes += numCols;
                rowOffsetsRes[r] = r * numCols;
            }
            rowOffsetsRes[numRows] = numRows * numCols;
        } else {
            // Common case: There are zeros and non-zeros in the arg.

            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, false);

            // Scan the argument's values again and only insert the non-zeros into the result.
            valuesArg = arg->getValues();
            VT *valuesRes = res->getValues();
            size_t *colIdxsRes = res->getColIdxs();
            size_t *rowOffsetsRes = res->getRowOffsets();
            size_t i = 0;
            rowOffsetsRes[0] = 0;
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    VT temp = valuesArg[c];
                    if (temp != 0) {
                        valuesRes[i] = temp;
                        colIdxsRes[i] = c;
                        i++;
                    }
                }
                valuesArg += arg->getRowSkip();
                rowOffsetsRes[r + 1] = i;
            }
        }
    }
};

// ----------------------------------------------------------------------------
//  CSRMatrix  <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTres, typename VTarg> class CastObj<CSRMatrix<VTres>, CSRMatrix<VTarg>> {

  public:
    static void apply(CSRMatrix<VTres> *&res, const CSRMatrix<VTarg> *arg, DCTX(ctx)) {
        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VTres>>(arg->getNumCols(), arg->getNumRows(),
                                                              arg->getNumNonZeros(), true);

        auto res_val = res->getValues();
        auto res_cidx = res->getColIdxs();
        auto res_roff = res->getRowOffsets();

        auto arg_val = arg->getValues();
        auto arg_cidx = arg->getColIdxs();
        auto arg_roff = arg->getRowOffsets();

        for (size_t nz = 0; nz < arg->getNumNonZeros(); nz++) {
            res_val[nz] = static_cast<VTres>(arg_val[nz]);
            res_cidx[nz] = arg_cidx[nz];
        }

        for (size_t r = 0; r < arg->getNumRows() + 1; r++) {
            res_roff[r] = arg_roff[r];
        }
    }
};

// ----------------------------------------------------------------------------
//  Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> class CastObj<Matrix<VTRes>, Matrix<VTArg>> {

  public:
    static void apply(Matrix<VTRes> *&res, const Matrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, static_cast<VTRes>(arg->get(r, c)));
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
//  Column <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<Column<VT>, DenseMatrix<VT>> {

  public:
    static void apply(Column<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numCols == 1) {
            // The input matrix has a single column.
            const size_t rowSkipArg = arg->getRowSkip();
            if (rowSkipArg == 1) {
                // The input's single column is stored contiguously.
                // Reuse the input's memory for the result (zero-copy).
                res = DataObjectFactory::create<Column<VT>>(numRows, arg->getValuesSharedPtr());
            } else {
                // The input's single column is not stored contiguosly.
                // Copy the input data to the result.
                res = DataObjectFactory::create<Column<VT>>(numRows, false);
                const VT *valuesArg = arg->getValues();
                VT *valuesRes = res->getValues();
                for (size_t r = 0; r < numRows; r++) {
                    valuesRes[r] = *valuesArg;
                    valuesArg += rowSkipArg;
                }
            }
        } else {
            // The input matrix has zero or multiple columns.
            throw std::runtime_error("CastObj::apply: cannot cast a matrix with zero or mutliple columns to Column");
        }
    }
};

// ----------------------------------------------------------------------------
//  DenseMatrix <- Column
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<DenseMatrix<VT>, Column<VT>> {

  public:
    static void apply(DenseMatrix<VT> *&res, const Column<VT> *arg, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<VT>>(arg->getNumRows(), 1, arg->getValuesSharedPtr());
    }
};

// ----------------------------------------------------------------------------
//  Column <- Frame
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<Column<VT>, Frame> {

  public:
    static void apply(Column<VT> *&res, const Frame *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        if (numCols == 1 && arg->getColumnType(0) == ValueTypeUtils::codeFor<VT>) {
            // The input frame has a single column of the result's value type.
            // Zero-cost cast from frame to Column.
            // TODO This case could even be used for (un)signed integers of the
            // same width, involving a reinterpret cast of the pointers.
            // TODO Can we avoid this const_cast?
            res = DataObjectFactory::create<Column<VT>>(numRows, arg->getColumn<VT>(0)->getValuesSharedPtr());
        } else {
            // The input frame has multiple columns and/or other value types
            // than the result.
            throw std::runtime_error("CastObj::apply: cannot cast Frame with mutliple columns to Column");
        }
    }
};

// ----------------------------------------------------------------------------
//  Frame <- Column
// ----------------------------------------------------------------------------

template <typename VT> class CastObj<Frame, Column<VT>> {

  public:
    static void apply(Frame *&res, const Column<VT> *arg, DCTX(ctx)) {
        std::vector<Structure *> colMats;
        DenseMatrix<VT> *argMat = nullptr;
        castObj<DenseMatrix<VT>>(argMat, arg, ctx);
        colMats.push_back(argMat);
        res = DataObjectFactory::create<Frame>(colMats, nullptr);
    }
};