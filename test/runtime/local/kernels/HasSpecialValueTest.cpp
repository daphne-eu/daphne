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

#include <catch.hpp>
#include <cmath>
#include <tags.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/HasSpecialValue.h>

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue integer", TAG_KERNELS, (DenseMatrix, CSRMatrix), (int32_t, uint32_t, size_t)) {

    using DT = TestType;

    auto mat = genGivenVals<DT>(3, {
        0, 1, 3,
        4, 5, 6,
        7, 8, 9
    });

    SECTION("hasSpecialValue") {
        CHECK_FALSE(hasSpecialValue(mat));
    }


}

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue floating point", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double_t)) {

    using DT = TestType;

    auto sigNaN = std::numeric_limits<double_t>::signaling_NaN();
    auto quietNaN = std::numeric_limits<double_t>::quiet_NaN();
    auto inf = std::numeric_limits<double_t>::infinity();

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
        CHECK(hasSpecialValue(sigNaNMat));
    }

    SECTION("quiet NaN"){
        CHECK(hasSpecialValue(quietNaNMat));
    }

    SECTION("infinity") {
        CHECK(hasSpecialValue(infinityMat));
    }
}
