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
#include <runtime/local/kernels/OneHot.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <stdexcept>

#define DATA_TYPES DenseMatrix
#define VALUE_TYPES int64_t, double

TEMPLATE_PRODUCT_TEST_CASE("OneHot", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTArg = TestType;
    using VT = typename DTArg::VT;
    using DTRes = DTArg;
    
    auto * arg = genGivenVals<DTArg>(3, {
        -1,  0, 1,
        -10, 1, VT(1.5),
        100, 2, 1,
    });

    DenseMatrix<int64_t> * info = nullptr;
    DTRes * res = nullptr;

    SECTION("normal encoding") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {-1, 3, 2});
        auto * exp = genGivenVals<DTRes>(3, {
            -1,  1, 0, 0, 0, 1,
            -10, 0, 1, 0, 0, 1,
            100, 0, 0, 1, 0, 1
        });

        oneHot(res, arg, info, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp, res);
    }
    SECTION("normal encoding - skip columns") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {0, 0, 3});
        auto * exp = genGivenVals<DTRes>(3, {
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        });

        oneHot(res, arg, info, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp, res);
    }
    SECTION("negative example - invalid info shape (not row matrix)") {
        info = genGivenVals<DenseMatrix<int64_t>>(3, {-1, 3, 2});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::runtime_error);
    }
    SECTION("negative example - invalid info shape (too small)") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {-1, 3});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::runtime_error);
    }
    SECTION("negative example - invalid info value (int < -1)") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {-2, 3, 2});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::runtime_error);
    }
    SECTION("negative example - empty selection") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {0, 0, 0});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::runtime_error);
    }
    SECTION("negative example - not enough space reserved (0 <= info value < arg value)") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {-1, 2, 2});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::out_of_range);
    }
    SECTION("negative example - out of bounds (arg value negative)") {
        info = genGivenVals<DenseMatrix<int64_t>>(1, {3, 3, 3});
        REQUIRE_THROWS_AS(oneHot(res, arg, info, nullptr), std::out_of_range);
    }

    DataObjectFactory::destroy(arg, info);
}