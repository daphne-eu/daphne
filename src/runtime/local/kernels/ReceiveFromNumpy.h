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
struct ReceiveFromNumpy {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void receiveFromNumpy(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpy<DTRes>::apply(res, upper, lower, size, ctx);
}



// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

struct NoOpDeleter {
    void operator()(double* p) {
        // don't delete p because the memory comes from numpy
    }
    void operator()(float* p){}
    void operator()(int32_t* p){}
    void operator()(int8_t* p){}
    void operator()(int64_t* p){}
    void operator()(uint64_t* p){}
    void operator()(uint32_t* p){}
    void operator()(uint8_t* p){}
};

template<typename VT>
struct ReceiveFromNumpy<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<VT>>(size, size, std::shared_ptr<VT[]>((VT*)((upper<<32)|lower), NoOpDeleter()));
    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H