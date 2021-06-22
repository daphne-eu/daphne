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
      const size_t numElements = numRows * numCols;
      const size_t rowSkip = arg->getRowSkip();
      const VT *valuesBegin = arg->getValues();

      std::priority_queue<uint32_t> pQueue;

      uint32_t hashedValueOut = 0;

      size_t rowIdx = 0;
      size_t colIdx = 0;
      size_t elementIdx = 0;

      // Initialize KVM with K values.
      for (; rowIdx < numRows; rowIdx++) {
          for (; colIdx < numCols && elementIdx < K; colIdx++, elementIdx++) {

              const VT *el = valuesBegin + rowSkip * rowIdx + colIdx;

              MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);
              pQueue.push(hashedValueOut);
          }
      }

      //for (size_t idx = 0; idx < K; idx++) {
      //    const VT *el = valuesBegin + idx;
    
      //    MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);
      //    pQueue.push(hashedValueOut);
      //}

      for (; rowIdx < numRows; rowIdx++) {
          for (; colIdx < numCols; colIdx++) {

              const VT *el = valuesBegin + rowSkip * rowIdx + colIdx;
              MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);

              if (hashedValueOut < pQueue.top()) {
                  pQueue.pop();
                  pQueue.push(hashedValueOut);
              }
          }
      }


      //for (size_t idx = K; idx < numElements; idx++) {
      //    const VT *el = valuesBegin + idx;
      //    MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValueOut);

      //    if (hashedValueOut < pQueue.top()) {
      //        pQueue.pop();
      //        pQueue.push(hashedValueOut);
      //    }
      //}

      size_t kMinVal = pQueue.top();
      const size_t maxVal = std::numeric_limits<std::uint32_t>::max();
      double kMinValNormed =
          static_cast<double>(kMinVal) / static_cast<double>(maxVal);
  
      printf("kMinValNormed %f\n", kMinValNormed);
      printf("result %f\n", (K - 1) / kMinValNormed);

      return (K - 1) / kMinValNormed;
  }
};

template <typename VT> struct NumDistinctApprox<CSRMatrix<VT>> {
    static size_t apply(const CSRMatrix<VT> *arg, size_t K, uint64_t seed = 1234567890) {

        const size_t numElements = arg->getNumRows() * arg->getNumCols();
        const VT* valuesBegin = arg->getValues();
        const size_t numNonZeros = arg->getNumNonZeros();

        std::priority_queue<uint32_t> pQueue;

        uint32_t hashedValue = 0;

        // Initialize KVM with K values.
        // Inserting first zero if one exists.
        if (numElements != numNonZeros) {
            const VT zero = 0;
            MurmurHash3_x86_32(&zero, sizeof(VT), seed, &hashedValue);
            pQueue.push(hashedValue);
        }

        for (size_t idx = 1; idx < K; idx++) {
            const VT *el = valuesBegin + idx;
    
            MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValue);
            pQueue.push(hashedValue);
        }

        for (size_t idx = K; idx < numNonZeros; idx++) {
            const VT *el = valuesBegin + idx;
            MurmurHash3_x86_32(el, sizeof(VT), seed, &hashedValue);
    
            if (hashedValue < pQueue.top()) {
                pQueue.pop();
                pQueue.push(hashedValue);
            }
        }

      size_t kMinVal = pQueue.top();
      const size_t maxVal = std::numeric_limits<std::uint32_t>::max();
      double kMinValNormed =
          static_cast<double>(kMinVal) / static_cast<double>(maxVal);

      return (K - 1) / kMinValNormed;
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_NUMDISTINCTAPPROX_H
