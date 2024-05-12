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
#include <runtime/local/kernels/CondMatMatSca.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "CondMatMatSca (" opName ")"
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES int64_t, double

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT * argCond = nullptr;
    DT * argThen = nullptr;
    VT argElse;
    DT * exp = nullptr;

    SECTION("example 1") {
        argCond = genGivenVals<DT>(3, {
            true, false, false,
            false, true, false,
            false, false, true
        });
        argThen = genGivenVals<DT>(3, {
            VT(1.5), 2, 3,
            4,       5, 6,
            7,       8, 9
        });

        argElse = VT(-1.5);

        exp = genGivenVals<DT>(3, {
            VT(1.5), argElse, argElse,
            argElse, 5,       argElse,
            argElse, argElse, 9
        });
    }

    DT * res = nullptr;
    condMatMatSca(res, argCond, argThen, argElse, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(argCond, argThen, exp, res);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("invalid shape"), TAG_KERNELS, (DATA_TYPES), (int64_t)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT * argCond = genGivenVals<DT>(3, {
        true, false, false,
        false, true, false,
        false, false, true
    });
    
    DT * argThen = nullptr;
    VT argElse;

    SECTION("then matrix too small") {
        argThen = genGivenVals<DT>(2, {
            VT(1.5), 2, 3,
            4,       5, 6
        });

        argElse = VT(-1.5);
    }

    DT * res = nullptr;

    REQUIRE_THROWS_AS(condMatMatSca(res, argCond, argThen, argElse, nullptr), std::runtime_error);

    DataObjectFactory::destroy(argCond, argThen);
    if (res != nullptr)
        DataObjectFactory::destroy(res);
}