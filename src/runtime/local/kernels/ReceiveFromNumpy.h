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
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};

template<class DTRes>
struct ReceiveFromNumpyF32 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
template<class DTRes>
struct ReceiveFromNumpyI8 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
template<class DTRes>
struct ReceiveFromNumpyI32 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
template<class DTRes>
struct ReceiveFromNumpyI64 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};

template<class DTRes>
struct ReceiveFromNumpyUI8 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
template<class DTRes>
struct ReceiveFromNumpyUI32 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
template<class DTRes>
struct ReceiveFromNumpyUI64 {
    static void apply(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) = delete;
};
// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes>
void receiveFromNumpyDouble(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyDouble<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyF32(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyF32<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyI8(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyI8<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyI32(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyI32<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyI64(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyI64<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyUI8(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyUI8<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyUI32(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyUI32<DTRes>::apply(res, upper, lower, size, ctx);
}
template<class DTRes>
void receiveFromNumpyUI64(DTRes *& res,  int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
    ReceiveFromNumpyUI64<DTRes>::apply(res, upper, lower, size, ctx);
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

template<>
struct ReceiveFromNumpyDouble<DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<double>>(size, size, std::shared_ptr<double[]>((double*)((upper<<32)|lower), NoOpDeleter()));
    }
};

template<>
struct ReceiveFromNumpyF32<DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<float>>(size, size, std::shared_ptr<float[]>((float*)((upper<<32)|lower), NoOpDeleter()));
    }
};

template<>
struct ReceiveFromNumpyI8<DenseMatrix<int8_t>> {
    static void apply(DenseMatrix<int8_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<int8_t>>(size, size, std::shared_ptr<int8_t[]>((int8_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};


template<>
struct ReceiveFromNumpyI32<DenseMatrix<int32_t>> {
    static void apply(DenseMatrix<int32_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<int32_t>>(size, size, std::shared_ptr<int32_t[]>((int32_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};

template<>
struct ReceiveFromNumpyI64<DenseMatrix<int64_t>> {
    static void apply(DenseMatrix<int64_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<int64_t>>(size, size, std::shared_ptr<int64_t[]>((int64_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};

template<>
struct ReceiveFromNumpyUI8<DenseMatrix<uint8_t>> {
    static void apply(DenseMatrix<uint8_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<uint8_t>>(size, size, std::shared_ptr<uint8_t[]>((uint8_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};


template<>
struct ReceiveFromNumpyUI32<DenseMatrix<uint32_t>> {
    static void apply(DenseMatrix<uint32_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<uint32_t>>(size, size, std::shared_ptr<uint32_t[]>((uint32_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};

template<>
struct ReceiveFromNumpyUI64<DenseMatrix<uint64_t>> {
    static void apply(DenseMatrix<uint64_t> *& res, int64_t upper, int64_t lower, int64_t size, DCTX(ctx)) {
        res = DataObjectFactory::create<DenseMatrix<uint64_t>>(size, size, std::shared_ptr<uint64_t[]>((uint64_t*)((upper<<32)|lower), NoOpDeleter()));
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_RECEIVEFROMNUMPY_H