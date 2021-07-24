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


#include <cstddef>
#include <cstdlib>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/NumDistinctApprox.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <stdexcept>
#include <tags.h>
#include <catch.hpp>

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {

    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numElements = 10000;
    std::srand(123456789);
    size_t expectedNumDistinct = 0;
    size_t approxResult = 0;


    SECTION("numDistinctApprox distinct") {

        std::vector<VT> v(numElements,0);
        std::generate_n(v.begin(), numElements/100, std::rand);

        auto mat10000 = genGivenVals<DT>(100, v);
        approxResult = numDistinctApprox(mat10000, 64);
        expectedNumDistinct = 100;
    }

    SECTION("numDistinctApprox distinct leading 100 zeros") {

        std::vector<VT> v(numElements,0);
        std::srand(123456789);

        auto it = v.begin();
        std::advance(it, 100);
        std::generate_n(it, numElements/100, std::rand);

        auto matZerosAtStart = genGivenVals<DT>(100, v);
        approxResult = numDistinctApprox(matZerosAtStart, 64);
        expectedNumDistinct = 100;
    }

    SECTION("numDistinctApprox #distinct elements < K") {

        std::vector<VT> v(numElements, 0);
        v[0] = VT(1);
        auto twoDistinctValsMat = genGivenVals<DT>(100, v);

        approxResult = numDistinctApprox(twoDistinctValsMat, 64);
        expectedNumDistinct = 2;

    }

    // Allow +/-10% error. When error is bigger something is either
    // wrong parametriced (K to small) or the algorithm broke.
    const size_t tolerance = static_cast<size_t>(expectedNumDistinct * 0.1);
    CHECK(approxResult >= expectedNumDistinct - tolerance);
    CHECK(approxResult <= expectedNumDistinct + tolerance);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - Dense-Submatrix", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {

    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numRows = 100;
    const size_t numCols = 100;
    const size_t numElements = numRows * numCols;
    size_t expectedNumDistinct = 0;
    size_t approxResult = 0;

    std::vector<VT> v(numElements,0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);

    SECTION("numDistinctApprox for Sub-DenseMatrix full matrix - sanity check") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows,
            0,
            numCols
        );

        approxResult = numDistinctApprox(fullSubMat, 64);
        expectedNumDistinct = numElements;
    }

    SECTION("numDistinctApprox for Sub-DenseMatrix") {
        auto subMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows/100,
            0,
            numCols
        );

        approxResult = numDistinctApprox(subMat, 64);
        expectedNumDistinct = numElements/100;

    }

    SECTION("numDistinctApprox for Sub-DenseMatrix #distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows/100,
            0,
            numCols/10
        );

        approxResult = numDistinctApprox(smallSubMat, 64);
        expectedNumDistinct = numElements/1000;
    }

    // Allow +/-10% error. When error is bigger something is either
    // wrong parametriced (K to small) or the algorithm broke.
    const size_t tolerance = static_cast<size_t>(expectedNumDistinct * 0.1);
    CHECK(approxResult >= expectedNumDistinct - tolerance);
    CHECK(approxResult <= expectedNumDistinct + tolerance);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - CSR-Submatrix", TAG_KERNELS, (CSRMatrix), (double, uint32_t)) {

    using DT = TestType;
    using VT = typename DT::VT;

    const size_t numRows = 100;
    const size_t numCols = 100;
    const size_t numElements = numRows * numCols;
    size_t expectedNumDistinct = 0;
    size_t approxResult = 0;

    std::vector<VT> v(numElements, 0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(numRows, v);


    SECTION("numDistinctApprox for Sub-CSRMatrix full matrix - sanity check") {
        auto fullSubMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows
        );
        approxResult = numDistinctApprox(fullSubMat, 64);
        expectedNumDistinct = numElements;
    }

    SECTION("numDistinctApprox Sub-CSRMatrix") {
        auto subMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows/100
        );
        approxResult = numDistinctApprox(subMat, 64);
        expectedNumDistinct = numElements/100;
    }

    SECTION("numDistinctApprox for Sub-CSRMatrix #distinct elements < K") {
        auto smallSubMat = DataObjectFactory::create<DT>(mat10000,
            0,
            numRows/100
        );

        approxResult = numDistinctApprox(smallSubMat, 128);
        expectedNumDistinct = numElements/100;
    }

    // Allow +/-10% error. When error is bigger something is either
    // wrong parametriced (K to small) or the algorithm broke.
    const size_t tolerance = static_cast<size_t>(expectedNumDistinct * 0.1);
    CHECK(approxResult >= expectedNumDistinct - tolerance);
    CHECK(approxResult <= expectedNumDistinct + tolerance);
}
