/*
 * Copyright 2024 The DAPHNE Consortium
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

#include <catch.hpp>
#include <tags.h>

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AvgPoolBackward.h>
#include <runtime/local/kernels/MaxPoolBackward.h>

template <typename DT> DT *genInput() {
    return genGivenVals<DT>(
        2, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,

            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54});
}

template <typename DT> DT *genDOut() {
    return genGivenVals<DT>(2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

template <class DT>
void check_max(const DT *in, const DT *dOut, const DT *exp,
               DaphneContext *dctx) {
    DT *res = nullptr;
    MaxPoolBackward<DT, DT>::apply(res, in, dOut, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1,
                                   dctx);
    CHECK(*res == *exp);
}

template <class DT>
void check_avg(const DT *in, const DT *dOut, const DT *exp,
               DaphneContext *dctx) {
    DT *res = nullptr;
    AvgPoolBackward<DT, DT>::apply(res, in, dOut, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1,
                                   dctx);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("pool_bwd_avg", TAG_DNN, (DenseMatrix),
                           (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    auto dctx = setupContextAndLogger();

    auto inputs = genInput<DT>();
    auto dOut = genDOut<DT>();

    auto out = genGivenVals<DT>(
        2, {0.25, 0.50, 0.50, 0.75, 1.00, 1.00, 0.75, 1.00, 1.00,
            1.25, 1.50, 1.50, 1.75, 2.00, 2.00, 1.75, 2.00, 2.00,
            2.25, 2.50, 2.50, 2.75, 3.00, 3.00, 2.75, 3.00, 3.00,

            0.25, 0.50, 0.50, 0.75, 1.00, 1.00, 0.75, 1.00, 1.00,
            1.25, 1.50, 1.50, 1.75, 2.00, 2.00, 1.75, 2.00, 2.00,
            2.25, 2.50, 2.50, 2.75, 3.00, 3.00, 2.75, 3.00, 3.00});

    check_avg(inputs, dOut, out, dctx.get());

    DataObjectFactory::destroy(inputs);
    DataObjectFactory::destroy(dOut);
    DataObjectFactory::destroy(out);
}

TEMPLATE_PRODUCT_TEST_CASE("pool_bwd_max", TAG_DNN, (DenseMatrix),
                           (float, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;

    auto dctx = setupContextAndLogger();

    auto inputs = genInput<DT>();
    auto dOut = genDOut<DT>();

    auto out = genGivenVals<DT>(2, {1, 0, 2, 0, 0, 0, 3,  0, 4, 5, 0,  6, 0,  0,
                                    0, 7, 0, 8, 9, 0, 10, 0, 0, 0, 11, 0, 12,

                                    1, 0, 2, 0, 0, 0, 3,  0, 4, 5, 0,  6, 0,  0,
                                    0, 7, 0, 8, 9, 0, 10, 0, 0, 0, 11, 0, 12});

    check_max(inputs, dOut, out, dctx.get());

    DataObjectFactory::destroy(inputs);
    DataObjectFactory::destroy(dOut);
    DataObjectFactory::destroy(out);
}
