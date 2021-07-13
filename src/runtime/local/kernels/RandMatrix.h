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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H
#define SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <random>
#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, typename VTArg>
struct RandMatrix {
    static void apply(DTRes *& res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, typename VTArg>
void randMatrix(DTRes *& res, size_t numRows, size_t numCols, VTArg min, VTArg max, double sparsity, int64_t seed) {
    RandMatrix<DTRes, VTArg>::apply(res, numRows, numCols, min, max, sparsity, seed);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct RandMatrix<DenseMatrix<VT>, VT> {
    static void apply(DenseMatrix<VT> *& res, size_t numRows, size_t numCols, VT min, VT max, double sparsity, int64_t seed) {
        assert(numRows > 0 && "numRows must be > 0");
        assert(numCols > 0 && "numCols must be > 0");
        assert(min <= max && "min must be <= max");
        assert(sparsity >= 0.0 && sparsity <= 1.0 &&
               "sparsity has to be in the interval [0.0, 1.0]");

        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        if (seed == -1) {
            std::random_device rd;
            std::uniform_int_distribution<int64_t> seedRnd;
            seed = seedRnd(rd);
        }

        std::mt19937 genVal(seed);
        std::mt19937 genSparse(seed * 3);
        
        static_assert(
                std::is_floating_point<VT>::value || std::is_integral<VT>::value,
                "the value type must be either floating point or integral"
        );
        typename std::conditional<
                std::is_floating_point<VT>::value,
                std::uniform_real_distribution<VT>,
                std::uniform_int_distribution<VT>
        >::type distrVal(min, max);
        std::uniform_real_distribution<double> distrSparse(0.0, 1.0);

        VT * valuesRes = res->getValues();
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                if (distrSparse(genSparse) > sparsity)
                    valuesRes[c] = VT(0);
                else
                    valuesRes[c] = distrVal(genVal);
            }
            valuesRes += res->getRowSkip();
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RANDMATRIX_H