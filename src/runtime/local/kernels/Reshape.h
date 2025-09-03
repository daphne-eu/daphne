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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H
#define SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <stdexcept>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct Reshape {
    static void apply(DTRes *&res, const DTArg *arg, size_t numRows, size_t numCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg>
void reshape(DTRes *&res, const DTArg *arg, size_t numRows, size_t numCols, DCTX(ctx)) {
    Reshape<DTRes, DTArg>::apply(res, arg, numRows, numCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Reshape<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, size_t numRowsRes, size_t numColsRes,
                      DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        // If the result and argument shapes are the same, return the argument (no need to create a new object).
        if (numRowsRes == numRowsArg && numColsRes == numColsArg) {
            arg->increaseRefCounter();
            res = const_cast<DenseMatrix<VT> *>(arg);
            return;
        }

        if (numRowsRes * numColsRes != numRowsArg * numColsArg)
            throw std::runtime_error("reshape must retain the number of cells");

        if (arg->getRowSkip() == numColsArg || numRowsArg == 1) {
            // The data in arg is contiguous in memory. The result can share the data with the argument.

            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numColsRes, arg->getValuesSharedPtr());
        } else {
            // The data in arg is not contiguous in memory. We must copy the data row by row from the argument to the
            // result.

            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numColsRes, false);

            auto resVals = res->getValues();
            auto argVals = arg->getValues();
            size_t numArgRows = arg->getNumRows();
            size_t numArgCols = arg->getNumCols();
            for (size_t r = 0; r < numArgRows; r++) {
                std::copy(argVals, argVals + numArgCols, resVals);
                argVals += arg->getRowSkip();
                resVals += numArgCols;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Reshape<CSRMatrix<VT>, CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const CSRMatrix<VT> *arg, size_t numRowsRes, size_t numColsRes, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        // If the result and argument shapes are the same, return the argument (no need to create a new object).
        if (numRowsRes == numRowsArg && numColsRes == numColsArg) {
            arg->increaseRefCounter();
            res = const_cast<CSRMatrix<VT> *>(arg);
            return;
        }

        if (numRowsRes * numColsRes != numRowsArg * numColsArg)
            throw std::runtime_error("reshape must retain the number of cells");

        // In general, the non-zero values in the result are the same as in the argument. Ideally, the result simply
        // reuses the values array of the argument (no need to copy or rearrange the values array). We only need to map
        // the colIdxs and rowOffsets. However, special attention is needed when the argument is a view into a larger
        // CSRMatrix.

        // Create the result matrix and set the values array.
        if (!arg->isView()) {
            // The argument is not a view into a larger CSRMatrix. The result can share the values array with the
            // argument.
            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(numRowsRes, numColsRes, arg->getNumNonZeros(),
                                                               arg->getValuesSharedPtr());
            // TODO Should we ever pass an existing data object for the result, we must make sure that either (a) the
            // values array of the argument is copied, or (b) the values array of the result is destroyed and replaced
            // by the argument's values array.
        } else {
            // The argument is a view into a larger CSRMatrix. In theory, the result could share the values array with
            // the argument. However, due to the way we represent views in CSRMatrix, it would be a bit involved. Thus,
            // we resort to copying the value array for simplicity.
            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(numRowsRes, numColsRes, arg->getNumNonZeros(), false);
            std::copy(arg->getValues(0), arg->getValues(0) + arg->getNumNonZeros(), res->getValues());
        }

        // Set the colIdxs and rowOffsets arrays.
        // We iterate over all rows and in each row over all non-zero values. For each non-zero, we calculate its
        // linear position in the logical matrix from its rowIdx and colIdx in the argument. Then, we calculate the
        // rowIdx and colIdx in the result from this linear position.
        size_t *rowOffsetsRes = res->getRowOffsets();
        size_t *colIdxsRes = res->getColIdxs();
        const size_t *rowOffsetsArg = arg->getRowOffsets();
        *rowOffsetsRes++ = 0; // the values of the 1st row always start at position 0 in the values and colIdxs arrays
        size_t rResPrev = 0;
        size_t numNonZerosProcessed = 0;
        for (size_t rArg = 0; rArg < numRowsArg; rArg++) {
            size_t numNonZerosRowArg = rowOffsetsArg[rArg + 1] - rowOffsetsArg[rArg];
            const size_t *colIdxsArg = arg->getColIdxs(rArg);
            for (size_t iArg = 0; iArg < numNonZerosRowArg; iArg++) {
                size_t cArg = colIdxsArg[iArg];
                size_t pos = rArg * numColsArg + cArg;
                size_t rRes = pos / numColsRes;
                size_t cRes = pos % numColsRes;
                for (size_t k = rResPrev; k < rRes; k++) // there could have been empty rows since the previous non-zero
                    *rowOffsetsRes++ = numNonZerosProcessed;
                *colIdxsRes++ = cRes;
                rResPrev = rRes;
                numNonZerosProcessed++;
            }
        }
        for (size_t k = rResPrev; k < numRowsRes; k++) // there could have been empty rows since the previous non-zero
            *rowOffsetsRes++ = numNonZerosProcessed;
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct Reshape<Matrix<VT>, Matrix<VT>> {
    static void apply(Matrix<VT> *&res, const Matrix<VT> *arg, size_t numRowsRes, size_t numColsRes, DCTX(ctx)) {
        const size_t numRowsArg = arg->getNumRows();
        const size_t numColsArg = arg->getNumCols();

        // If the result and argument shapes are the same, return the argument (no need to create a new object).
        if (numRowsRes == numRowsArg && numColsRes == numColsArg) {
            arg->increaseRefCounter();
            res = const_cast<Matrix<VT> *>(arg);
            return;
        }

        if (numRowsRes * numColsRes != numRowsArg * numColsArg)
            throw std::runtime_error("Reshape: new shape must retain the number of cells");

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRowsRes, numColsRes, false);

        res->prepareAppend();
        for (size_t r = 0, rArg = 0, cArg = 0; r < numRowsRes; ++r)
            for (size_t c = 0; c < numColsRes; ++c) {
                res->append(r, c, arg->get(rArg, cArg++));
                cArg = (cArg != numColsArg) * cArg;
                rArg += (cArg == 0);
            }
        res->finishAppend();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_RESHAPE_H