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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <algorithm>
#include <random>
#include <set>
#include <type_traits>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes>
struct ReceiveFromNumpyDouble {
    static void apply(DTRes *& res, int64_t arg, int64_t size, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void receiveFromNumpyDouble(DTRes *& res, int64_t arg, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyDouble<DTRes>::apply(res, arg, size, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template<>
struct ReceiveFromNumpyDouble<DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, intptr_t arg, int64_t size, DCTX(ctx)) {
        printf("%ld", arg);
        res = DataObjectFactory::create<DenseMatrix<double>>(size, size, (double* )arg);
        res -> print(std::cout);
    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H