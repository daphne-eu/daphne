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
#include <runtime/local/datastructures/Matrix.h>

#include <cmath>
#include <cstddef>
#include <cstdio>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg, typename VTVal> struct HasSpecialValue {
    static bool apply(const DTArg *arg, VTVal val, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks each element of the matrix against a value `val`. Returns `true` upon finding the first matching
 * element in the matrix. This operation is nan-aware, i.e., if nan is given as the special value to search, the
 * elements in the matrix will be checked for nan.
 *
 * @param arg the matrix to check
 * @param val the special value to search
 * @return `true` if `val` is contained in `arg`; `false`, otherwise
 */
template <class DTArg, typename VTVal> bool hasSpecialValue(const DTArg *arg, VTVal val, DCTX(ctx)) {
    return HasSpecialValue<DTArg, VTVal>::apply(arg, val, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// bool <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VT> struct HasSpecialValue<DenseMatrix<VT>, VT> {
    static bool apply(const DenseMatrix<VT> *arg, VT val, DCTX(ctx)) {
        auto numRows = arg->getNumRows();
        auto numCols = arg->getNumCols();

        if (std::isnan(val))
            for (auto rowIdx = 0ul; rowIdx < numRows; rowIdx++)
                for (auto colIdx = 0ul; colIdx < numCols; colIdx++) {
                    if (std::isnan(arg->get(rowIdx, colIdx)))
                        return true;
                }
        else
            for (auto rowIdx = 0ul; rowIdx < numRows; rowIdx++)
                for (auto colIdx = 0ul; colIdx < numCols; colIdx++)
                    if (arg->get(rowIdx, colIdx) == val)
                        return true;

        return false;
    }
};

// ----------------------------------------------------------------------------
// bool <- CSRMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VT> struct HasSpecialValue<CSRMatrix<VT>, VT> {
    static bool apply(const CSRMatrix<VT> *arg, VT val, DCTX(ctx)) {
        auto numRows = arg->getNumRows();
        auto numCols = arg->getNumCols();
        auto numNonZeros = arg->getNumNonZeros();
        auto numElements = numRows * numCols;
        auto vBegin = arg->getValues(0);
        auto vEnd = arg->getValues(numRows);
        auto hasZeroes = numNonZeros < numElements;
        auto zero = VT(0);

        if (std::isnan(val)) {
            for (auto it = vBegin; it != vEnd; it++)
                if (std::isnan(*it))
                    return true;
        } else {
            if (hasZeroes && zero == val) // test zero
                return true;
            for (auto it = vBegin; it != vEnd; it++)
                if (*it == val)
                    return true;
        }
        return false;
    }
};

// ----------------------------------------------------------------------------
// bool <- Matrix, scalar
// ----------------------------------------------------------------------------

template <typename VT> struct HasSpecialValue<Matrix<VT>, VT> {
    static bool apply(const Matrix<VT> *arg, VT val, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (std::isnan(val)) {
            for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
                for (size_t colIdx = 0; colIdx < numCols; ++colIdx)
                    if (std::isnan(arg->get(rowIdx, colIdx)))
                        return true;
        } else {
            for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
                for (size_t colIdx = 0; colIdx < numCols; ++colIdx)
                    if (arg->get(rowIdx, colIdx) == val)
                        return true;
        }

        return false;
    }
};
