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

template <class DTArg> struct MatSymCheck {
  static bool apply(const DTArg *arg) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> bool matSymCheck(const DTArg *arg) {
  return MatSymCheck<DTArg>::apply(arg);
}

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

template <typename VT> struct MatSymCheck<DenseMatrix<VT>> {
  static bool apply(const DenseMatrix<VT> *arg) {

    const size_t numRows = arg->getNumRows();
    const size_t numCols = arg->getNumCols();

    if (numRows != numCols || numRows <= 1 || numCols <= 1) {
      throw std::runtime_error("Provided matrix is not square.");
    }

    size_t totalCheckToMake = (numRows * numRows - numRows) / 2;
    size_t rowIdx = 0;
    size_t checksInRow = numRows - 1;

    const VT *start = arg->getValues();
    const VT *fin = start + ((numCols * numCols) - 2); // last element to check

    while (totalCheckToMake > 0) {

      const VT *pt1 = start + (rowIdx * numRows) + (numRows - checksInRow);
      const VT *pt2 = start + (numRows - checksInRow) * numRows + rowIdx;

      if (*pt1 != *pt2) {
        return false;
      }

      checksInRow--;
      totalCheckToMake--;

      if (checksInRow == 0) {
        rowIdx++;
        checksInRow = numRows - (rowIdx + 1);
      }
    }

    return true;
  }
};

template <typename VT> struct MatSymCheck<CSRMatrix<VT>> {
  static bool apply(const CSRMatrix<VT> *arg) {

    const size_t numRows = arg->getNumRows();
    const size_t numCols = arg->getNumCols();
    const VT *values = arg->getValues();

    if (numRows != numCols || numRows <= 1 || numCols <= 1) {
      throw std::runtime_error("Provided matrix is not square.");
    }

    return false;
    // throw std::string("Not implemented");
  }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_MATSYMCHECK_H
