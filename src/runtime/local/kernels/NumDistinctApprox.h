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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_NUMDISTINCTAPPROX_H
#define SRC_RUNTIME_LOCAL_KERNELS_NUMDISTINCTAPPROX_H

#include <bits/stdint-uintn.h>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <queue>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <util/UniqueBoundedPriorityQueue.h>
#include <tuple>
#include <util/MurmurHash3.h>
#include <vector>
template <class DTArg> struct NumDistinctApprox {
    static size_t apply(const DTArg *arg, size_t K) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Approximates the number of distinct values using K-Minimum Values.
 * Uses the 32-bit MurmurHash3 hashing algorithm.
 */
template <class DTArg> size_t numDistinctApprox(const DTArg *arg, size_t K) {
    return NumDistinctApprox<DTArg>::apply(arg, K);
}

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

template <typename VT> struct NumDistinctApprox<DenseMatrix<VT>> {
  static size_t apply(const DenseMatrix<VT> *arg, size_t K, uint64_t seed = 1234567890) {

      const size_t numRows = arg->getNumRows();
      const size_t numCols = arg->getNumCols();

      UniqueBoundedPriorityQueue<uint32_t> pQueue(K);

      uint32_t hashedValueOut = 0;

      for(auto rowIdx = 0; rowIdx < numRows; rowIdx++) {
          for(auto colIdx = 0; colIdx < numCols; colIdx++) {
              auto el = arg->get(rowIdx, colIdx);
              MurmurHash3_x86_32(&el, sizeof(VT), seed, &hashedValueOut);
              pQueue.push(hashedValueOut);
          }
      }

      size_t kMinVal = pQueue.top();
      const size_t maxVal = std::numeric_limits<std::uint32_t>::max();
      double kMinValNormed =
          static_cast<double>(kMinVal) / static_cast<double>(maxVal);
  
      return (K - 1) / kMinValNormed;
  }
};

template <typename VT> struct NumDistinctApprox<CSRMatrix<VT>> {
    static size_t apply(const CSRMatrix<VT> *arg, size_t K, uint64_t seed = 1234567890) {

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const size_t numElements = numRows * numCols;
        const VT* values = arg->getValues();
        const size_t* rowOffsets = arg->getRowOffsets();
        const VT zero = 0;

        UniqueBoundedPriorityQueue<uint32_t> pQueue(K);
        uint32_t hashedValueOut = 0;

        size_t rowIdx = 0;
        size_t colIdx = 0;
        size_t elementIdx = 0;
        bool stopInit = false;
        const size_t * colIdxBegin = arg->getColIdxs(rowIdx);
        const size_t * colIdxEnd = arg->getColIdxs(rowIdx+1);

      for(size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
          for(size_t colIdx = 0; colIdx < numCols; colIdx++) {
              VT el = arg->get(rowIdx, colIdx);
              MurmurHash3_x86_32(&el, sizeof(VT), seed, &hashedValueOut);
              pQueue.push(hashedValueOut);
          }
      }
        /*
        while( rowIdx < numRows ) {
            colIdx = 0;
            colIdxBegin = arg->getColIdxs(rowIdx);
            colIdxEnd = arg->getColIdxs(rowIdx+1);

            while( colIdx < numCols) {

                const size_t * ptrExpected = std::lower_bound(colIdxBegin, colIdxEnd, colIdx);
                const VT * el;

                // Check wether this colIdx does exist in this row's colIdxs.
                if(ptrExpected == colIdxEnd || *ptrExpected != colIdx) {
                    el = &zero;
                } else {
                    el = (values + rowOffsets[rowIdx]) + (ptrExpected - colIdxBegin);
                }

                MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);
                pQueue.push(hashedValueOut);

                elementIdx++;

                if (elementIdx >= K) {
                    stopInit = true;
                    break;
                }
                colIdx++;
            }

            if (stopInit) {
                break;
            }
            rowIdx++;
        }


        // rowIdx/colIdx are now at the element where we left off.
        while ( rowIdx < numRows) {
            colIdxBegin = arg->getColIdxs(rowIdx);
            colIdxEnd = arg->getColIdxs(rowIdx+1);

            while ( colIdx < numCols) {

                const size_t * ptrExpected = std::lower_bound(colIdxBegin, colIdxEnd, colIdx);
                const VT * el;

                // Check wether this colIdx does exist in this row's colIdxs.
                if(ptrExpected == colIdxEnd || *ptrExpected != colIdx) {
                    el = &zero;
                } else {
                    el = (values + rowOffsets[rowIdx]) + (ptrExpected - colIdxBegin);
                }

                MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);

                if (hashedValueOut < pQueue.top()) {
                    pQueue.pop();
                    pQueue.push(hashedValueOut);
                }

                colIdx++;
            }

            colIdx = 0;
            rowIdx++;
       }

       */
      size_t kMinVal = pQueue.top();
      const size_t maxVal = std::numeric_limits<std::uint32_t>::max();
      double kMinValNormed =
          static_cast<double>(kMinVal) / static_cast<double>(maxVal);

      return (K - 1) / kMinValNormed;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_NUMDISTINCTAPPROX_H
