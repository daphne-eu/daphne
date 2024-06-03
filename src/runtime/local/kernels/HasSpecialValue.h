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
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cmath>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

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

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Bool <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Bool <- CSRMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VT, typename TestType> struct HasSpecialValue<CSRMatrix<VT>, TestType> {
    static bool apply(const CSRMatrix<VT> *arg, TestType testVal, DCTX(ctx)) {
        auto numRows = arg->getNumRows();
        auto numCols = arg->getNumCols();
        auto numNonZeros = arg->getNumNonZeros();
        auto numElements = numRows*numCols;
        auto vBegin = arg->getValues(0);
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

// ----------------------------------------------------------------------------
// Bool <- Matrix, scalar
// ----------------------------------------------------------------------------

template <typename VT, typename TestType> struct HasSpecialValue<Matrix<VT>, TestType> {
    static bool apply(const Matrix<VT> *arg, TestType testVal, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (std::isnan(testVal)) {
            for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
                for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                    const VT val = arg->get(rowIdx, colIdx);
                    if (std::isnan(val)) {
                        return true;
                    }
                }
            }
        } else {
            for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
                for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                    const VT val = arg->get(rowIdx, colIdx);
                    if (val == testVal) {
                        return true;
                    }
                }
            }
        }

        return false;
    }
};

// ----------------------------------------------------------------------------
// Bool <- ContiguousTensor, scalar
// ----------------------------------------------------------------------------

template <typename VT, typename TestType> struct HasSpecialValue<ContiguousTensor<VT>, TestType> {
    static bool apply(const ContiguousTensor<VT> *arg, TestType testVal, DCTX(ctx)) {
        if (std::isnan(testVal)) {
            for (size_t i=0; i < arg->total_element_count; i++) {
                if (std::isnan(arg->data[i])) {
                    return true;
                }
            }
        } else {
            for (size_t i=0; i < arg->total_element_count; i++) {
                if (arg->data[i] == testVal) {
                    return true;
                }
            }
        }

        return false;
    }
};

// ----------------------------------------------------------------------------
// Bool <- ChunkedTensor, scalar
// ----------------------------------------------------------------------------

template <typename VT, typename TestType> struct HasSpecialValue<ChunkedTensor<VT>, TestType> {
    static bool apply(const ChunkedTensor<VT> *arg, TestType testVal, DCTX(ctx)) {
        bool is_nan = std::isnan(testVal);

        for (size_t i=0; i < arg->total_chunk_count; i++) {
            auto current_chunk_ids = arg->getChunkIdsFromLinearChunkId(i);

            VT* current_chunk_ptr = arg->getPtrToChunk(current_chunk_ids);

            if (!arg->isPartialChunk(current_chunk_ids)) {
                for (size_t j=0; j < arg->chunk_element_count; j++) {
                    if (is_nan ? std::isnan(current_chunk_ptr[j]) : current_chunk_ptr[j] == testVal) {
                        return true;
                    }
                }
            } else {
                auto valid_id_bounds = arg->GetIdBoundsOfPartialChunk(current_chunk_ids);
                auto strides = arg->GetStridesOfPartialChunk(valid_id_bounds);
                auto valid_element_count = arg->GetElementCountOfPartialChunk(valid_id_bounds);

                for (size_t i=0; i < valid_element_count; i++) {
                    std::vector<size_t> element_ids;
                    element_ids.resize(arg->rank);

                    int64_t tmp = i;
                    for (int64_t j=arg->rank-1; j >= 0; j--) {
                        element_ids[j] = tmp / strides[j];
                        tmp = tmp % strides[j];
                    }
                    size_t lin_id = 0;
                    for (size_t j = 0; j < arg->rank; j++) {
                        lin_id += element_ids[j] * strides[j]; 
                    }

                    if (is_nan ? std::isnan(current_chunk_ptr[lin_id]) : current_chunk_ptr[lin_id] == testVal) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
};
