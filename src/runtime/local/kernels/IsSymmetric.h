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

        // TODO add cache-conscious operations
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

                const VT val1 = arg->get(rowIdx, colIdx);
                const VT val2 = arg->get(colIdx, rowIdx);

                if (val1 != val2) {
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

        if (numRows != numCols) {
            throw std::runtime_error("Provided matrix is not square.");
        }

        // singular matrix is considered symmetric.
        if (numRows <= 1 || numCols <= 1) {
            return true;
        }

        const size_t* rowOffsets = arg->getRowOffsets();

        std::vector<size_t> positions(numRows, 0);

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {

            const VT* rowA = arg->getValues(rowIdx);
            const size_t* colIdxsA = arg->getColIdxs(rowIdx);
            const size_t numNonZerosA = arg->getNumNonZeros(rowIdx);

            size_t posA = positions[rowIdx];
            size_t probedColIdxA = -1;

            // Only get idx when one exists.
            if (posA < numNonZerosA) {
                probedColIdxA = colIdxsA[posA];
            }

            // positions contains idx of diagonal element idx, try to advance.
            if (probedColIdxA == rowIdx && posA < numNonZerosA) {
                positions[rowIdx]++;
            }

            for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

                posA = positions[rowIdx];

                VT valA = 0;
                probedColIdxA = -1;

                // Only get idx when one exists.
                if (posA < numNonZerosA) {
                    probedColIdxA = colIdxsA[posA];
                }

                // index exists element not zero
                if (colIdx == probedColIdxA && posA < numNonZerosA) {
                    valA = rowA[posA];
                    // Advance to next 'unused' position.
                    positions[rowIdx]++;
                }

                // B references the transposed element to compare for symmetrie.
                const VT* rowB = arg->getValues(colIdx);
                const size_t* colIdxsB = arg->getColIdxs(colIdx);
                const size_t numNonZerosB = arg->getNumNonZeros(colIdx);

                const size_t posB = positions[colIdx];
                size_t probedColIdxB = -1;

                // Only get idx when one exists.
                if (posB < numNonZerosB) {
                    probedColIdxB = colIdxsB[posB];
                }

                VT valB = 0;
                // Index exists element not zero.
                if (rowIdx == probedColIdxB && posB < numNonZerosA) {
                    valB = rowB[posB];
                    // Advance to next 'unused' position.
                    positions[colIdx]++;
                }

                if (valA != valB) {
                    return false;
                }
            }
        }

        return true;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_ISSYMMETRIC_H
