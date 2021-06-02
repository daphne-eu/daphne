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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_ISSYMMETRIC_H
#define SRC_RUNTIME_LOCAL_KERNELS_ISSYMMETRIC_H

#include <cstddef>
#include <cstdio>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <string>

template <class DTArg> struct IsSymmetric {
    static bool apply(const DTArg *arg) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> bool isSymmetric(const DTArg *arg) {
    return IsSymmetric<DTArg>::apply(arg);
}

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

/**
 * @brief Checks for symmetrie of a `DenseMatrix`.
 *
 * Checks for symmetrie in a DenseMatrix. Returning early if a check failes, or
 * the matrix is not square. Singular matrixes are considered square. 
 */

template <typename VT> struct IsSymmetric<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> *arg) {

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t rowSkip = arg->getRowSkip();

        const VT* values = arg->getValues();

        if (numRows != numCols) {
            throw std::runtime_error("Provided matrix is not square.");
        }

        // singular matrix is considered symmetric.
        if (numRows <= 1 || numCols <= 1) {
            return true;
        }

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

                const VT* val1 = values + rowSkip *  rowIdx +  colIdx;
                const VT* val2 = values + rowSkip * colIdx + rowIdx;

                if (*val1 != *val2) {
                    return false;
                }
            }
        }
        return true;
    }
};

template <typename VT> struct IsSymmetric<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> *arg) {

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const VT zero = 0;
        const VT* values = arg->getValues();
        const size_t* rowOffsets = arg->getRowOffsets();

        if (numRows != numCols) {
            throw std::runtime_error("Provided matrix is not square.");
        }

        // singular matrix is considered symmetric.
        if (numRows <= 1 || numCols <= 1) {
            return true;
        }

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {

            const size_t * colIdxVal1Begin = arg->getColIdxs(rowIdx);
            const size_t * colIdxVal1End = arg->getColIdxs(rowIdx+1); // first index of another row.

            for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

                const size_t * ptrExpected1 = std::lower_bound(colIdxVal1Begin, colIdxVal1End, colIdx);
                const VT * val1;

                if(ptrExpected1 == colIdxVal1End || *ptrExpected1 != colIdx) {
                    val1 = &zero;
                } else {
                    val1 = (values + rowOffsets[rowIdx]) + (ptrExpected1 - colIdxVal1Begin);
                }

                const size_t * colIdxVal2Begin = arg->getColIdxs(colIdx);
                const size_t * colIdxVal2End = arg->getColIdxs(colIdx+1);
                const size_t * ptrExpected2 = std::lower_bound(colIdxVal2Begin, colIdxVal2End, rowIdx);
                const VT * val2;

                if(ptrExpected2 == colIdxVal2End || *ptrExpected2 != rowIdx) {
                    val2 = &zero;
                } else {
                    val2 = (values + rowOffsets[colIdx]) + (ptrExpected2 - colIdxVal2Begin);
                }

                if (*val1 != *val2) {
                    return false;
                }
            }
        }
        return true;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_ISSYMMETRIC_H
