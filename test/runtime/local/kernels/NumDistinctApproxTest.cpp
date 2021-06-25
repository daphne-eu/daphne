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


#include <cstdlib>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/NumDistinctApprox.h>
#include <stdexcept>
#include <tags.h>
#include <catch.hpp>

TEMPLATE_PRODUCT_TEST_CASE("numDistinctApprox", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {

    using DT = TestType;

    const size_t numElements = 10000;
    std::vector<typename DT::VT> v(numElements,0);
    std::srand(123456789);

    std::generate_n(v.begin(), numElements/100, std::rand);
    auto mat10000 = genGivenVals<DT>(100, v);

    SECTION("numDistinctApprox distinct") {

        // Allow +/-10% error. When error is bigger something is either
        // wrong parametriced (K to small) or the algorithm broke.
        auto approxResult = numDistinctApprox(mat10000, 64);
        auto isResultBelow20PercentOff = approxResult <= 110 && approxResult >= 90;
        CHECK(isResultBelow20PercentOff);
    }
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

    auto sanityCheckSubMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows(),
        0,
        mat10000->getNumCols()
    );

    SECTION("numDistinctApprox for Sub-DenseMatrix") {

        // Allow +/-10% error. When error is bigger something is either
        // wrong parametriced (K to small) or the algorithm broke.
        auto approxResult = numDistinctApprox(subMat, 64);
        auto isResultBelow20PercentOff = approxResult <= 110 && approxResult >= 90;
        CHECK(isResultBelow20PercentOff);

        auto sanityCheckResult = numDistinctApprox(sanityCheckSubMat, 64);
        isResultBelow20PercentOff = sanityCheckResult <= 110 && sanityCheckResult >= 90;
        CHECK_FALSE(isResultBelow20PercentOff);
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

    auto sanityCheckSubMat = DataObjectFactory::create<DT>(mat10000,
        0,
        mat10000->getNumRows()
    );

    SECTION("numDistinctApprox Sub-CSRMatrix") {

        // Allow +/-20% error. When error is bigger something is either
        // wrong parametriced (K to small) or the algorithm broke.
        auto approxResult = numDistinctApprox(subMat, 64);
        auto isResultBelow20PercentOff = approxResult <= 120 && approxResult >= 80;
        CHECK(isResultBelow20PercentOff);

        auto sanityCheckResult = numDistinctApprox(sanityCheckSubMat, 64);
        isResultBelow20PercentOff = sanityCheckResult <= 110 && sanityCheckResult >= 90;
        CHECK_FALSE(isResultBelow20PercentOff);
    }

}
