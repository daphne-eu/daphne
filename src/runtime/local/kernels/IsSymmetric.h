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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MATSYMCHECK_H
#define SRC_RUNTIME_LOCAL_KERNELS_MATSYMCHECK_H

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
 * Checks for symmetrie in a DenseMatrix. Returning early if a check failes, or the matrix is not
 * square. Singular matrixes are considered square. The maximum amount of required checks is 
 * (#row * #rows - #rows)/2 elements.
*/

template <typename VT> struct IsSymmetric<DenseMatrix<VT>> {
  static bool apply(const DenseMatrix<VT> *arg) {

    const size_t numRows = arg->getNumRows();
    const size_t numCols = arg->getNumCols();

    if (numRows != numCols ) {
      throw std::runtime_error("Provided matrix is not square.");
    }

    // singular matrix is considered symmetric.
    if(numRows <= 1 || numCols <= 1) {
        return true;
    }

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
        for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

            if (arg->get(colIdx, rowIdx) != arg->get(rowIdx, colIdx)) {
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

    if (numRows != numCols ) {
      throw std::runtime_error("Provided matrix is not square.");
    }

    // singular matrix is considered symmetric.
    if(numRows <= 1 || numCols <= 1) {
        return true;
    }

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
        for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {

            if (arg->get(colIdx, rowIdx) != arg->get(rowIdx, colIdx)) {
                return false;
            }
        }
    }
    return true;
  }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_MATSYMCHECK_H
