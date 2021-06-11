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
#ifndef SRC_RUNTIME_LOCAL_KERNELS_HASSPECIALVALUE_H
#define SRC_RUNTIME_LOCAL_KERNELS_HASSPECIALVALUE_H

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <string>
#include <cmath>
#include <type_traits>

template <class DTArg, class DTFArg> struct HasSpecialValue {
    static bool apply(const DTArg *arg, bool (*specValTesttestFunc)(DTFArg)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks each element of the matrix against a provided function
 * returning a boolean. Stops early when the test function returns true.
 *
 * Example use
 *
 *  bool isNaN(double val) {
 *      return std::isnan(val);
 *  }
 *
 *  bool hasSpecVal = hasSpecialValue(mat, isNaN);
 *
 * @param
 * @param arg Pointer to a matrix.
 * @param testFunc Pointer to a bool returning function taking the matrix element type as an argument.
 * @return Returns true when the test function returns true.
 */
template <class DTArg, class TestFunc> bool hasSpecialValue(const DTArg *arg, TestFunc testFunc) {
    return HasSpecialValue<DTArg, TestFunc>::apply(arg, testFunc);
}

template <typename VT, typename TestFunc> struct HasSpecialValue<DenseMatrix<VT>, TestFunc> {
    static bool apply(const DenseMatrix<VT> *arg, TestFunc testFunc) {

        const VT* values = arg->getValues();
        size_t numRows = arg->getNumRows();
        size_t numCols = arg->getNumCols();
        size_t rowSkip = arg->getRowSkip();

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            for (size_t colIdx = 0; colIdx < numCols; colIdx++) {

                const VT* value = values + rowSkip * rowIdx + colIdx;

                if (testFunc(*value)) {
                    return true;
                }
            }
        }

        return false;
    }
};

template <typename VT, typename TestFunc> struct HasSpecialValue<CSRMatrix<VT>, TestFunc> {
    static bool apply(const CSRMatrix<VT> *arg, TestFunc testFunc) {

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const VT zero = 0;
        const VT* values = arg->getValues();
        const size_t* rowOffsets = arg->getRowOffsets();

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {

            const size_t * colIdxBegin = arg->getColIdxs(rowIdx);
            const size_t * colIdxEnd = arg->getColIdxs(rowIdx+1);

            for (size_t colIdx = 0; colIdx < numCols; colIdx++) {

                const size_t * ptrExpected = std::lower_bound(colIdxBegin, colIdxEnd, colIdx);
                const VT * value;

                // Check wether this colIdx does exist in this row's colIdxs.
                if(ptrExpected == colIdxEnd || *ptrExpected != colIdx) {
                    value = &zero;
                } else {
                    value = (values + rowOffsets[rowIdx]) + (ptrExpected - colIdxBegin);
                }

                if (testFunc(*value)) {
                    return true;
                }

            }
        }

        return false;
    }
};

#endif
