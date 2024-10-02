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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CondMatScaSca.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "CondMatScaSca (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int64_t, double

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *argCond = nullptr;
    VT argThen;
    VT argElse;
    DT *exp = nullptr;

    SECTION("example 1") {
        argCond = genGivenVals<DT>(3, {true, false, false, false, true, false, false, false, true});

        argThen = VT(1.5);

        argElse = VT(-1.5);

        exp = genGivenVals<DT>(3, {VT(1.5), argElse, argElse, argElse, VT(1.5), argElse, argElse, argElse, VT(1.5)});
    }

    DT *res = nullptr;
    condMatScaSca(res, argCond, argThen, argElse, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(argCond, exp, res);
}