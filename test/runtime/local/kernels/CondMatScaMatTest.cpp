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
#include <runtime/local/kernels/CondMatScaMat.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "CondMatScaMat (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int64_t, double

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *argCond = nullptr;
    VT argThen;
    DT *argElse = nullptr;
    DT *exp = nullptr;

    SECTION("example 1") {
        argCond = genGivenVals<DT>(3, {true, false, false, false, true, false, false, false, true});

        argThen = VT(-1.5);

        argElse = genGivenVals<DT>(3, {1, 2, 3, VT(4.5), 5, 6, 7, 8, 9});
        exp = genGivenVals<DT>(3, {argThen, 2, 3, VT(4.5), argThen, 6, 7, 8, argThen});
    }

    DT *res = nullptr;
    condMatScaMat(res, argCond, argThen, argElse, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(argCond, argElse, exp, res);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("invalid shape"), TAG_KERNELS, (DATA_TYPES), (int64_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *argCond = genGivenVals<DT>(3, {true, false, false, false, true, false, false, false, true});

    VT argThen;
    DT *argElse = nullptr;

    SECTION("else matrix too small") {
        argThen = VT(-1.5);

        argElse = genGivenVals<DT>(2, {1, 2, 3, VT(4.5), 5, 6});
    }

    DT *res = nullptr;

    REQUIRE_THROWS_AS(condMatScaMat(res, argCond, argThen, argElse, nullptr), std::runtime_error);

    DataObjectFactory::destroy(argCond, argElse);
    if (res != nullptr)
        DataObjectFactory::destroy(res);
}