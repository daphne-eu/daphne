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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/MatMul.h>
#include <runtime/local/kernels/Transpose.h>
#include <runtime/local/kernels/Syrk.h>
#include "run_tests.h"

#include <tags.h>

#include <catch.hpp>

#include <vector>

template<class DT>
void checkSyrk(const DT * arg, DCTX(dctx)) {
    DT * resExp = nullptr;
    DT * argT = nullptr;
    transpose(argT, arg, dctx);
    matMul(resExp, argT, arg, false, false, dctx);

    DT * resAct = nullptr;
    syrk(resAct, arg, nullptr);
    CHECK(*resAct == *resExp);
    DataObjectFactory::destroy(resAct);
    DataObjectFactory::destroy(resExp);
}

TEMPLATE_PRODUCT_TEST_CASE("Syrk", TAG_KERNELS, (DenseMatrix), (float, double)) {
    using DT = TestType;
    auto dctx = setupContextAndLogger();

    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        1, 2, 3,
        3, 1, 2,
        2, 3, 1,
    });
    auto m2 = genGivenVals<DT>(3, {
        13, 13, 10,
        10, 13, 13,
        13, 10, 13,
    });
    auto m3 = genGivenVals<DT>(2, {
        1, 0, 3, 0,
        0, 0, 2, 0,
    });
    auto m4 = genGivenVals<DT>(4, {
        0, 1,
        2, 0,
        1, 1,
        0, 0,
    });
    auto m5 = genGivenVals<DT>(2, {
        3, 4,
        2, 2,
    });
    auto v0 = genGivenVals<DT>(3, {
        0,
        0,
        0
    });
    auto v1 = genGivenVals<DT>(3, {
        1,
        1,
        1
    });
    auto v2 = genGivenVals<DT>(3, {
        1,
        2,
        3
    });

    checkSyrk(m0, dctx.get());
    checkSyrk(m1, dctx.get());
    checkSyrk(m2, dctx.get());
    checkSyrk(m3, dctx.get());
    checkSyrk(m4, dctx.get());
    checkSyrk(m5, dctx.get());
    checkSyrk(v0, dctx.get());
    checkSyrk(v1, dctx.get());
    checkSyrk(v2, dctx.get());

    DataObjectFactory::destroy(m0, m1, m2, m3, m4, m5, v0, v1, v2);
}