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

    // The new data generation.
    const size_t numRows = 100;
    const size_t numCols = 100;
    const double sparsity = 0.1;
    // Each of the following cases generates approximately 1000 non-zero values.
    // The test case is automatically run separately for each of the sections.
    DT * mat = nullptr;
    size_t expectedNumDistinct;
    auto approxResult = 0;
    SECTION("unique values") {
        randMatrix<DT, VT>(mat, numRows, numCols, VT(0), VT(100000000), sparsity, -1);
        expectedNumDistinct = 1000;
        approxResult = numDistinctApprox(mat, 128);
    }
    SECTION("many, non-unique values") { // more than k unique values
        randMatrix<DT, VT>(mat, numRows, numCols, VT(0), VT(200), sparsity, -1);
        expectedNumDistinct = 200;
        approxResult = numDistinctApprox(mat, 128);
    }
    /*
    SECTION("few, non-unique values") { // less than k unique values
        randMatrix<DT, VT>(mat, numRows, numCols, VT(0), VT(10), sparsity, -1);
        expectedNumDistinct = 10;
        approxResult = numDistinctApprox(mat, 128);
    }
    */

    // Allow +/-20% error. When error is bigger something is either
    // wrong parametriced (K to small) or the algorithm broke.
    //auto approxResult = numDistinctApprox(mat, 64);
    const size_t tolerance = static_cast<size_t>(expectedNumDistinct * 0.2);
    CHECK(approxResult >= expectedNumDistinct - tolerance);
    CHECK(approxResult <= expectedNumDistinct + tolerance);
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - Dense-Submatrix", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {

    using DT = TestType;

    const size_t numElements = 10000;
    std::vector<typename DT::VT> v(numElements,0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(100, v);

    auto subMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows()/100,
        0,
        mat10000->getNumCols()
    );

    auto fullSubMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows(),
        0,
        mat10000->getNumCols()
    );

    SECTION("numDistinctApprox for Sub-DenseMatrix") {

        // Allow +/-10% error. When error is bigger something is either
        // wrong parametriced (K to small) or the algorithm broke.
        auto approxResult = numDistinctApprox(subMat, 64);
        auto isResultBelow10PercentOff = approxResult <= 110 && approxResult >= 90;
        CHECK(isResultBelow10PercentOff);

        approxResult = numDistinctApprox(fullSubMat, 64);
        isResultBelow10PercentOff = approxResult <= 11000 && approxResult >= 9000;
        CHECK(isResultBelow10PercentOff);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox - CSR-Submatrix", TAG_KERNELS, (CSRMatrix), (double, uint32_t)) {

    using DT = TestType;

    const size_t numElements = 10000;
    std::vector<typename DT::VT> v(numElements,0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements, std::rand);
    auto mat10000 = genGivenVals<DT>(100, v);

    auto subMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows()/100
    );

    auto fullSubMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows()
    );

    SECTION("numDistinctApprox Sub-CSRMatrix") {

        // Allow +/-10% error. When error is bigger something is either
        // wrong parametriced (K to small) or the algorithm broke.
        auto approxResult = numDistinctApprox(subMat, 64);
        auto isResultBelow10PercentOff = approxResult <= 110 && approxResult >= 90;
        CHECK(isResultBelow10PercentOff);

        approxResult = numDistinctApprox(fullSubMat, 64);
        isResultBelow10PercentOff = approxResult <= 11000 && approxResult >= 9000;

        CHECK(isResultBelow10PercentOff);
    }

}
