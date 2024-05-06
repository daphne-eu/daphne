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

#include "run_tests.h"

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>

#ifdef USE_CUDA
    #include <runtime/local/kernels/CUDA/Pooling.h>
    #include "runtime/local/kernels/CUDA/CreateCUDAContext.h"
#else
    #include <runtime/local/kernels/Pooling.h>
#endif

#include <tags.h>

#include <catch.hpp>

#include <vector>

template<typename DT>
DT* genInput() {
    return genGivenVals<DT>(2, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
            122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
            142, 143, 144, 145, 146, 147, 148, 149, 150
    });
}

template<template<typename> class OP, class DT>
void check(const DT* in, const DT* exp, DaphneContext* dctx) {
    DT* res = nullptr;
    size_t out_h;
    size_t out_w;
#ifdef USE_CUDA
    CUDA::NN::Pooling::Forward<OP, DT, DT>::apply(res, out_h, out_w, in, in->getNumRows(), 3, 5, 5, 2, 2, 1, 1, 0, 0, dctx);
#else
    NN::Pooling::Forward<OP, DT, DT>::apply(res, out_h, out_w, in, in->getNumRows(), 3, 5, 5, 2, 2, 1, 1, 0, 0, dctx);
#endif
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("pool_fwd_avg", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    auto dctx = setupContextAndLogger();

    // two rgb "images" of 5x5 pixels
    auto inputs = genInput<DT>();

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    auto out_f2x2_s1x1_p0x0 = genGivenVals<DT>(2, {
            4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42,
                    44, 45, 46, 47, 54, 55, 56, 57, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 72,
            79, 80, 81, 82, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 104, 105, 106, 107, 109, 110, 111, 112, 114,
                    115, 116, 117, 119, 120, 121, 122, 129, 130, 131, 132, 134, 135, 136, 137, 139, 140, 141, 142, 144,
                    145, 146, 147
    });

    check<NN::Pooling::AVG>(inputs, out_f2x2_s1x1_p0x0, dctx.get());

    DataObjectFactory::destroy(inputs);
    DataObjectFactory::destroy(out_f2x2_s1x1_p0x0);
}

TEMPLATE_PRODUCT_TEST_CASE("pool_fwd_max", TAG_DNN, (DenseMatrix), (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    auto dctx = setupContextAndLogger();

#ifdef USE_CUDA
    CUDA::createCUDAContext(dctx.get());
#endif

    // two rgb "images" of 5x5 pixels
    auto inputs = genInput<DT>();

    // expected output when used with settings filter 2x2, stride 1x1, padding 0x0
    auto out_f2x2_s1x1_p0x0 = genGivenVals<DT>(2, {
            7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45,
                    47, 48, 49, 50, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 72, 73, 74, 75,
            82, 83, 84, 85, 87, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99, 100, 107, 108, 109, 110, 112, 113, 114, 115, 117,
                    118, 119, 120, 122, 123, 124, 125, 132, 133, 134, 135, 137, 138, 139, 140, 142, 143, 144, 145, 147,
                    148, 149, 150
    });

    check<NN::Pooling::MAX>(inputs, out_f2x2_s1x1_p0x0, dctx.get());

    DataObjectFactory::destroy(inputs);
    DataObjectFactory::destroy(out_f2x2_s1x1_p0x0);
}
