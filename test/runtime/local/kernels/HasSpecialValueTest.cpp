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

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/Tensor.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <catch.hpp>
#include <cmath>
#include <limits>
#include <tags.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/HasSpecialValue.h>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define TENSOR_TYPES ContiguousTensor, ChunkedTensor

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue - integer", TAG_KERNELS, (DATA_TYPES), (uint32_t)) {

    using DT = TestType;

    auto specialMat = genGivenVals<DT>(4, {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 0, 2,
        3, 4, 5, 1
    });

    auto nonSpecialMat = genGivenVals<DT>(3, {
        0, 0, 3,
        4, 5, 6,
        7, 8, 9
    });

    SECTION("hasSpecialValue check if test function is applied correctly.") {
        CHECK(hasSpecialValue(specialMat, typename DT::VT(1), nullptr));
        CHECK_FALSE(hasSpecialValue(nonSpecialMat, typename DT::VT(1), nullptr));
    }
}


TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue - DenseMatrix-Submatrix.", TAG_KERNELS, DenseMatrix, (uint32_t)) {

    using DT = TestType;

    auto specialMat = genGivenVals<DT>(4, {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 0, 2,
        3, 4, 5, 1
    });

    auto subNonSpecialMat = DataObjectFactory::create<DT>(specialMat,
        1,
        specialMat->getNumRows() - 1,
        1,
        specialMat->getNumCols() - 1
    );

    SECTION("hasSpecialValue for Sub-DenseMatrix") {
        CHECK(hasSpecialValue(specialMat, typename DT::VT(1), nullptr));
        CHECK_FALSE(hasSpecialValue(subNonSpecialMat, typename DT::VT(1), nullptr));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue - CSRMatrix-Submatrix.", TAG_KERNELS, CSRMatrix, (uint32_t)) {

    using DT = TestType;

    auto specialMat = genGivenVals<DT>(4, {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 0, 2,
        3, 4, 5, 1
    });

    auto subNonSpecialMat = DataObjectFactory::create<DT>(specialMat,
        1,
        specialMat->getNumRows() - 2
    );

    SECTION("hasSpecialValue for Sub-CSRMatrix") {
        CHECK(hasSpecialValue(specialMat, typename DT::VT(1), nullptr));
        CHECK_FALSE(hasSpecialValue(subNonSpecialMat, typename DT::VT(1), nullptr));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("hasSpecialValue - floating point", TAG_KERNELS, (DATA_TYPES), (double)) {

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

    SECTION("Check for special values std::isnan/std::isinf.") {
        CHECK(hasSpecialValue(sigNaNMat, sigNaN, nullptr));
        CHECK(hasSpecialValue(quietNaNMat, quietNaN, nullptr));
        CHECK(hasSpecialValue(infinityMat, inf, nullptr));
        CHECK_FALSE(hasSpecialValue(infinityMat, sigNaN, nullptr));
    }
}

TEST_CASE("HasSpecialValue - ContiguousTensor") {
    auto t_0 = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,44,2}), InitCode::ZERO);
    auto t_iota = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,44,2}), InitCode::IOTA);
    auto t_sig = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,44,2}), InitCode::IOTA);
    auto t_quiet = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,44,2}), InitCode::IOTA);
    auto t_inf = DataObjectFactory::create<ContiguousTensor<double>>(std::vector<size_t>({2,44,2}), InitCode::IOTA);


    REQUIRE(hasSpecialValue(t_0, 0, nullptr));
    REQUIRE(!hasSpecialValue(t_0, 1, nullptr));

    REQUIRE(hasSpecialValue(t_iota, 42, nullptr));
    REQUIRE(!hasSpecialValue(t_iota, -1.0, nullptr));

    t_sig->data[2] = std::numeric_limits<double>::signaling_NaN();
    t_quiet->data[3] = std::numeric_limits<double>::quiet_NaN();
    t_inf->data[0] =  std::numeric_limits<double>::infinity();
    
    REQUIRE(hasSpecialValue(t_sig, std::numeric_limits<double>::signaling_NaN(), nullptr));
    REQUIRE(!hasSpecialValue(t_sig, std::numeric_limits<double>::quiet_NaN(), nullptr));

    REQUIRE(!hasSpecialValue(t_quiet, std::numeric_limits<double>::signaling_NaN(), nullptr));
    REQUIRE(hasSpecialValue(t_quiet, std::numeric_limits<double>::quiet_NaN(), nullptr));

    REQUIRE(hasSpecialValue(t_inf, std::numeric_limits<double>::infinity(), nullptr));
    REQUIRE(!hasSpecialValue(t_iota, std::numeric_limits<double>::infinity(), nullptr));
}

TEST_CASE("HasSpecialValue - ChunkedTensor") {
    auto t_0 = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({2,44,2}), std::vector<size_t>({2,2,2}), InitCode::ZERO);
    auto t_iota = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({2,44,2}), std::vector<size_t>({2,2,2}), InitCode::IOTA);
    auto t_sig = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({2,44,2}), std::vector<size_t>({2,2,2}), InitCode::IOTA);
    auto t_quiet = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({2,44,2}), std::vector<size_t>({2,2,2}), InitCode::IOTA);
    auto t_inf = DataObjectFactory::create<ChunkedTensor<double>>(std::vector<size_t>({2,44,2}), std::vector<size_t>({2,2,2}), InitCode::IOTA);


    REQUIRE(hasSpecialValue(t_0, 0, nullptr));
    REQUIRE(!hasSpecialValue(t_0, 1, nullptr));

    REQUIRE(hasSpecialValue(t_iota, 42, nullptr));
    REQUIRE(!hasSpecialValue(t_iota, -1.0, nullptr));

    t_sig->data[2] = std::numeric_limits<double>::signaling_NaN();
    t_quiet->data[3] = std::numeric_limits<double>::quiet_NaN();
    t_inf->data[0] =  std::numeric_limits<double>::infinity();
    
    REQUIRE(hasSpecialValue(t_sig, std::numeric_limits<double>::signaling_NaN(), nullptr));
    REQUIRE(!hasSpecialValue(t_sig, std::numeric_limits<double>::quiet_NaN(), nullptr));

    REQUIRE(!hasSpecialValue(t_quiet, std::numeric_limits<double>::signaling_NaN(), nullptr));
    REQUIRE(hasSpecialValue(t_quiet, std::numeric_limits<double>::quiet_NaN(), nullptr));

    REQUIRE(hasSpecialValue(t_inf, std::numeric_limits<double>::infinity(), nullptr));
    REQUIRE(!hasSpecialValue(t_iota, std::numeric_limits<double>::infinity(), nullptr));
}
