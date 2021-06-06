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

#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <catch.hpp>
#include <cmath>
#include <tags.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/HasSpecialValue.h>

bool isNaN(double val) {
    return std::isnan(val);
}

bool isInf(double val) {
    return std::isinf(val);
}

bool isOne(uint32_t val) {
    return val == 1; 
}

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue integer", TAG_KERNELS, (DenseMatrix, CSRMatrix), (uint32_t)) {

    using DT = TestType;

    auto specialMat = genGivenVals<DT>(3, {
        0, 1, 3,
        4, 5, 6,
        7, 8, 9
    });

    auto nonSpecialMat = genGivenVals<DT>(3, {
        0, 0, 3,
        4, 5, 6,
        7, 8, 9
    });

    SECTION("hasSpecialValue") {
        CHECK(hasSpecialValue(specialMat, isOne));
        CHECK_FALSE(hasSpecialValue(nonSpecialMat, isOne));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue floating point", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double)) {

    using DT = TestType;

    auto sigNaN = std::numeric_limits<double>::signaling_NaN();
    auto quietNaN = std::numeric_limits<double>::quiet_NaN();
    auto inf = std::numeric_limits<double>::infinity();

    auto sigNaNMat= genGivenVals<DT>(3, {
        0, 1, 3,
        4, 5, 6,
        7, 8, sigNaN
    });

    auto quietNaNMat = genGivenVals<DT>(3, {
        0, 1, 3,
        4, 5, 6,
        7, 8, quietNaN
    });

    auto infinityMat = genGivenVals<DT>(3, {
        0, 1, 3,
        4, 5, 6,
        7, 8, inf
    });

    SECTION("signaling NaN") {
        CHECK(hasSpecialValue(sigNaNMat, isNaN));
    }

    SECTION("quiet NaN"){
        CHECK(hasSpecialValue(quietNaNMat, isNaN));
    }

    SECTION("infinity") {
        CHECK(hasSpecialValue(infinityMat, isInf));
    }

    SECTION("no special value found") {
        CHECK_FALSE(hasSpecialValue(infinityMat, isNaN));
    }
}
