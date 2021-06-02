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


#include <bits/stdint-uintn.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/IsSymmetric.h>
#include <stdexcept>
#include <tags.h>
#include <catch.hpp>

TEMPLATE_PRODUCT_TEST_CASE("isSymmetric", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {

    using DT = TestType;
  
    auto symMat = genGivenVals<DT>(4, {
        0, 1, 2, 3,
        1, 1, 0, 6,
        2, 0, 2, 7,
        3, 6, 7, 3
    });

    auto asymMat = genGivenVals<DT>(4, {
        0, 1, 2, 3,
        0, 1, 4, 6,
        2, 4, 2, 7,
        3, 6, 7, 3
    });

    auto nonSquareMat = genGivenVals<DT>(3, {
        0, 1, 2, 3,
        0, 1, 4, 6,
        2, 4, 2, 7
    });

    auto squareZeroMat = genGivenVals<DT>(4, {
        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 0, 0
    });

    auto singularMat = genGivenVals<DT>(1, {1});

    SECTION("isSymmetric") {

        CHECK(isSymmetric<DT>(symMat));

        CHECK_FALSE(isSymmetric<DT>(asymMat));

        CHECK_THROWS_AS(isSymmetric<DT>(nonSquareMat), std::runtime_error);

        CHECK(isSymmetric<DT>(squareZeroMat));

        CHECK(isSymmetric<DT>(singularMat));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("isSymmetric DenseMatrix specific", TAG_KERNELS, DenseMatrix, (double, uint32_t)) {

    using DT = TestType;

    DT * centerSymMat = genGivenVals<DT>(5, {
        1, 1, 0, 1, 8,
        2, 0, 1, 2, 8,
        3, 1, 0, 3, 8,
        4, 2, 3, 0, 8,
        5, 4, 0, 4, 8
    });

    DT * symSubMat = DataObjectFactory::create<DT>(centerSymMat,
        1,
        centerSymMat->getNumRows() - 1,
        1,
        centerSymMat->getNumCols() - 1
    );

    SECTION("isSymmetric") {
        CHECK(isSymmetric(symSubMat));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("isSymmetric CSRMatrix specific", TAG_KERNELS, CSRMatrix, (double, uint32_t)) {

    using DT = TestType;

    DT * centerSymMat = genGivenVals<DT>(5, {
         1, 0, 1,
         0, 1, 2,
         1, 0, 3,
         2, 3, 0,
         4, 0, 4
    });

    DT * symSubMat = DataObjectFactory::create<DT>(centerSymMat,
        1,
        centerSymMat->getNumRows() - 1
    );

    SECTION("isSymmetric") {
        CHECK(isSymmetric(symSubMat));
    }
}
