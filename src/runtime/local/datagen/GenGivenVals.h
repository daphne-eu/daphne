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

#ifndef SRC_RUNTIME_LOCAL_DATAGEN_GENGIVENVALS_H
#define SRC_RUNTIME_LOCAL_DATAGEN_GENGIVENVALS_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <vector>

#include <cassert>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DT>
struct GenGivenVals {
    static DT * generate(size_t numRows, const std::vector<typename DT::VT> & elements, size_t minNumNonZeros = 0) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief A very simple data generator which populates a matrix with the
 * elements of the given `std::vector`.
 * 
 * Meant only for small matrices, mainly as a utility for testing and
 * debugging. Note that it can easily be used with an initializer list as
 * follows:
 * 
 * ```c++
 * // Generates the matrix  3 1 4
 * //                       1 5 9
 * auto m = genGivenVals<DenseMatrix<double>>(2, {3, 1, 4, 1, 5, 9});
 * ```
 * 
 * @param numRows The number of rows.
 * @param elements The data elements to populate the matrix with. Their number
 * must be divisible by `numRows`.
 * @param minNumNonZeros The minimum number of non-zeros to reserve space for
 * in a sparse matrix.
 * @return A matrix of the specified data type `DT` containing the provided
 * data elements.
 */
template<class DT>
DT * genGivenVals(size_t numRows, const std::vector<typename DT::VT> & elements, size_t minNumNonZeros = 0) {
    return GenGivenVals<DT>::generate(numRows, elements, minNumNonZeros);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// This data generator is not meant to be efficient. Nevertheless, note that we
// do not use the generic `set`/`append` interface to matrices here since this
// generator is meant to be used for writing tests for, besides others, those
// generic interfaces.

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct GenGivenVals<DenseMatrix<VT>> {
    static DenseMatrix<VT> * generate(size_t numRows, const std::vector<VT> & elements, size_t minNumNonZeros = 0) {
        if(numRows == 0)
            // We could return a 0x0 matrix, but this is often not what we want.
            // In many (test) cases, we want a 0xm matrix, but the number of columns
            // cannot be inferred if there are no elements. In such cases, callers
            // should rather construct a 0xm matrix via DataObjectFactory::create().
            throw std::runtime_error("genGivenVals(): numRows must not be zero");

        const size_t numCells = elements.size();
        if(numCells % numRows)
            throw std::runtime_error(
                    "genGivenVals(): the number of given data elements must be "
                    "divisible by given number of rows"
            );
        const size_t numCols = numCells / numRows;
        auto res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        memcpy(res->getValues(), elements.data(), numCells * sizeof(VT));
        return res;
    }
};

template<>
struct GenGivenVals<DenseMatrix<const char*>> {
    static DenseMatrix<const char*> * generate(size_t numRows, const std::vector<const char*> & elements, size_t minNumNonZeros = 0) {
        const size_t numCells = elements.size();
        assert((numCells % numRows == 0) && "number of given data elements must be divisible by given number of rows");
        const size_t numCols = numCells / numRows;
        auto res = DataObjectFactory::create<DenseMatrix<const char*>>(numRows, numCols, false);
        res->prepareAppend();
        for(size_t r = 0; r < numRows; r++)
            for(size_t c = 0; c < numCols; c++)
                res->append(r, c, elements[r * res->getRowSkip() + c]);
        res->finishAppend();
        return res;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct GenGivenVals<CSRMatrix<VT>> {
    static CSRMatrix<VT> * generate(size_t numRows, const std::vector<VT> & elements, size_t minNumNonZeros = 0) {
        const size_t numCells = elements.size();
        assert((numCells % numRows == 0) && "number of given data elements must be divisible by given number of rows");
        const size_t numCols = numCells / numRows;
        size_t numNonZeros = 0;
        for(VT v : elements)
            if(v != VT(0))
                numNonZeros++;
        auto res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, std::max(numNonZeros, minNumNonZeros), false);
        VT * values = res->getValues();
        size_t * colIdxs = res->getColIdxs();
        size_t * rowOffsets = res->getRowOffsets();
        size_t pos = 0;
        size_t colIdx = 0;
        size_t rowIdx = 0;
        rowOffsets[0] = 0;
        for(VT v : elements) {
            if(v != VT(0)) {
                values[pos] = v;
                colIdxs[pos] = colIdx;
                pos++;
            }
            colIdx++;
            if(colIdx == numCols) {
                colIdx = 0;
                rowOffsets[rowIdx++ + 1] = pos;
            }
        }
        return res;
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template<typename VT>
struct GenGivenVals<Matrix<VT>> {
    static Matrix<VT> * generate(size_t numRows, const std::vector<VT> & elements, size_t minNumNonZeros = 0) {
        // this is to simplify generating test matrices for the "Matrix" kernel specializations
        return GenGivenVals<DenseMatrix<VT>>::generate(numRows, elements, minNumNonZeros);
    }
};

#endif //SRC_RUNTIME_LOCAL_DATAGEN_GENGIVENVALS_H