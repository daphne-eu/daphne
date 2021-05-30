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
#include <runtime/local/util/MurmurHash.h>
#include <vector>

template <class DTArg> struct NumDistinctApprox {
  static size_t apply(const DTArg *arg, size_t K) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Approximates the number of distinct values using K-Minimum Values.
 */
template <class DTArg> size_t numDistinctApprox(const DTArg *arg, size_t K) {
  return NumDistinctApprox<DTArg>::apply(arg, K);
}

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

template <typename VT> struct NumDistinctApprox<DenseMatrix<VT>> {
  static size_t apply(const DenseMatrix<VT> *arg, size_t K) {
      constexpr uint_least64_t seed = 1234567890;
    const size_t maxVal = std::numeric_limits<std::size_t>::max();
    std::priority_queue<uint64_t> pQueue;

    std::hash<VT> hashFunction;
    const size_t numElements = arg->getNumRows() * arg->getNumCols();
    const VT *valuesBegin = arg->getValues();
    VT biggestVal = 0;

    // initialize KVM
    for (size_t idx = 0; idx < K; idx++) {
      const VT *el = valuesBegin + idx;

      uint64_t hashedValue = MurmurHash::MurmurHash64A(el, sizeof(VT), seed);

      printf("unhashed dec %d float %f\n", *el);
      printf("hashed %d\n", hashedValue);
      pQueue.push(hashedValue);
    }

    for (size_t idx = K; idx < numElements; idx++) {
      const VT *el = valuesBegin + idx;

      uint64_t hashedValue = MurmurHash::MurmurHash64A(el, sizeof(VT), seed);

      if (hashedValue < pQueue.top()) {
        pQueue.pop();
        pQueue.push(hashedValue);
      }
    }

    size_t kMinVal = pQueue.top();
    double kMinValNormed =
        static_cast<double>(kMinVal) / static_cast<double>(maxVal);

    printf("kMinValNormed %f\n", kMinValNormed);
    printf("result %f\n", (K - 1) / kMinValNormed);

    if (kMinValNormed == 0) { // edge case only 0
      return 1;
    }

    return (K - 1) / kMinValNormed;
  }
};

template <typename VT> struct NumDistinctApprox<CSRMatrix<VT>> {
  static size_t apply(const CSRMatrix<VT> *arg, size_t K) { return 100; }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_NUMDISTINCTAPPROX_H
