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
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <string>
#include <cmath>
#include <type_traits>

template <class DTArg, typename TestType> struct HasSpecialValue {
    static bool apply(const DTArg *arg, TestType testVal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks each element of the matrix against a value testVal. Returns
 * true oppon finding the first matching element in the matrix.
 *
 * @param
 * @param arg Pointer to a matrix.
 * @param testVal The value to test for in the matrix.
 * @return Returns true when finding a matchin element.
 */
template <class DTArg, typename TestType> bool hasSpecialValue(const DTArg *arg, TestType testVal, DCTX(ctx)) { 
    return HasSpecialValue<DTArg, TestType>::apply(arg, testVal, ctx);
}

template <typename VT, typename TestType> struct HasSpecialValue<DenseMatrix<VT>, TestType> {
    static bool apply(const DenseMatrix<VT> *arg, TestType testVal, DCTX(ctx)) {
        auto numRows = arg->getNumRows();
        auto numCols = arg->getNumCols();

        if(std::isnan(testVal)) {
            for(auto rowIdx = 0ul; rowIdx < numRows; rowIdx++) {
                for(auto colIdx = 0ul; colIdx < numCols; colIdx++) {
                    auto val = arg->get(rowIdx, colIdx);
                    if (std::isnan(val)) {
                        return true;
                    }
                }
            }
        } else {
            for(auto rowIdx = 0ul; rowIdx < numRows; rowIdx++) {
                for(auto colIdx = 0ul; colIdx < numCols; colIdx++) {
                    auto val = arg->get(rowIdx, colIdx);
                    if (val == testVal) {
                        return true;
                    }
                }
            }
        }

        return false;
    }
};

template <typename VT, typename TestType> struct HasSpecialValue<CSRMatrix<VT>, TestType> {
    static bool apply(const CSRMatrix<VT> *arg, TestType testVal, DCTX(ctx)) {
        auto numRows = arg->getNumRows();
        auto numCols = arg->getNumCols();
        auto numNonZeros = arg->getNumNonZeros();
        auto numElements = numRows*numCols;
        auto vBegin = arg->getRowValues(0);
        auto vEnd = arg->getValues(numRows);
        auto hasZeroes = numNonZeros < numElements;
        auto zero = VT(0);

        if(std::isnan(testVal)) {
            for(auto it = vBegin; it != vEnd; it++) {
                if (std::isnan(*it)) {
                    return true;
                }
            }
        } else {
            if (hasZeroes) { // test zero;
                if ((zero) == testVal) {
                    return true;
                }
            }
            for(auto it = vBegin; it != vEnd; it++) {
                if ((*it) == testVal) {
                    return true;
                }
            }
        }
        return false;
    }
};
