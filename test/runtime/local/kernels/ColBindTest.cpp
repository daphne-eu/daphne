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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColBind.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("ColBind", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        1, 2,
        3, 4,
        5, 6,
    });
    auto m1 = genGivenVals<DT>(3, {
       10, 20, 30,
       40, 50, 60,
       70, 80, 90,
    });
    
    auto exp = genGivenVals<DT>(3, {
       1, 2, 10, 20, 30,
       3, 4, 40, 50, 60,
       5, 6, 70, 80, 90,
    });
    
    DT * res = nullptr;
    colBind<DT, DT, DT>(res, m0, m1, nullptr);
    CHECK(*res == *exp);
    
    DataObjectFactory::destroy(m0, m1, exp, res);
}

TEST_CASE("ColBind - Frame", TAG_KERNELS) {
    auto c0 = genGivenVals<DenseMatrix<double>>(3, {1, 2, 3});
    auto c1 = genGivenVals<DenseMatrix<double>>(3, {4, 5, 6});
    auto c2 = genGivenVals<DenseMatrix<int64_t>>(3, {10, 20, 30});
    auto c3 = genGivenVals<DenseMatrix<int64_t>>(3, {40, 50, 60});
    auto c4 = genGivenVals<DenseMatrix<int64_t>>(3, {70, 80, 90});
    
    std::string l0 = "a";
    std::string l1 = "b";
    std::string l2 = "x";
    std::string l3 = "y";
    std::string l4 = "z";
    
    std::vector<Structure *> cols01 = {c0, c1};
    std::string labels01[] = {l0, l1};
    auto f01 = DataObjectFactory::create<Frame>(cols01, labels01);
    
    Frame* f234{};
    Frame* res{};
    SECTION("unique labels") {
        std::vector<Structure *> cols234 = {c2, c3, c4};
        std::string labels234[] = {l2, l3, l4};
        f234 = DataObjectFactory::create<Frame>(cols234, labels234);
        colBind<Frame, Frame, Frame>(res, f01, f234, nullptr);
        // Check dimensions.
        REQUIRE(res->getNumRows() == 3);
        REQUIRE(res->getNumCols() == 5);
        // Check column types.
        const std::string * labelsRes = res->getLabels();
        CHECK(labelsRes[0] == l0);
        CHECK(labelsRes[1] == l1);
        CHECK(labelsRes[2] == l2);
        CHECK(labelsRes[3] == l3);
        CHECK(labelsRes[4] == l4);
        // Check column labels.
        const ValueTypeCode * schemaRes = res->getSchema();
        CHECK(schemaRes[0] == ValueTypeCode::F64);
        CHECK(schemaRes[1] == ValueTypeCode::F64);
        CHECK(schemaRes[2] == ValueTypeCode::SI64);
        CHECK(schemaRes[3] == ValueTypeCode::SI64);
        CHECK(schemaRes[4] == ValueTypeCode::SI64);
        // Check column data.
        CHECK(*(res->getColumn<double>(0)) == *c0);
        CHECK(*(res->getColumn<double>(1)) == *c1);
        CHECK(*(res->getColumn<int64_t>(2)) == *c2);
        CHECK(*(res->getColumn<int64_t>(3)) == *c3);
        CHECK(*(res->getColumn<int64_t>(4)) == *c4);
        
        DataObjectFactory::destroy(res);
    }
    SECTION("non-unique labels") {
        std::vector<Structure *> cols1 = {c2, c3, c4};
        std::string labels1[] = {l0, l1, "c"};
        f234 = DataObjectFactory::create<Frame>(cols1, labels1);
        CHECK_THROWS(colBind<Frame, Frame, Frame>(res, f01, f234, nullptr));
    }
    
    DataObjectFactory::destroy(c0, c1, c2, c3, c4, f01, f234);
}

TEMPLATE_PRODUCT_TEST_CASE("ColBind", TAG_KERNELS, (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    size_t numRows = 4;
    size_t numCols = 5;
    size_t numNonZeros = 5;

    auto m0 = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, true);
    m0->set(1,1, VT(11.));
    m0->set(2,1, VT(21.));
    m0->set(2,3, VT(23.));
    m0->set(3,3, VT(33.));
    m0->set(3,4, VT(34.));
    
    DT * res = nullptr;
    
    SECTION("Normal on normal") {
        auto m1 = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, 3, true);
        m1->set(1,0, VT(10.));
        m1->set(1,2, VT(12.));
        m1->set(3,0, VT(30.));

        auto exp = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols*2, 8, true);
        exp->set(1,1, VT(11.));
        exp->set(2,1, VT(21.));
        exp->set(2,3, VT(23.));
        exp->set(3,3, VT(33.));
        exp->set(3,4, VT(34.));

        exp->set(1,0+numCols, VT(10.));
        exp->set(1,2+numCols, VT(12.));
        exp->set(3,0+numCols, VT(30.));

        colBind<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("View on normal") {
        size_t lowerBound = 1;
        size_t upperBound = 3;
        size_t rowsTake = upperBound - lowerBound;
        
        auto m1 = DataObjectFactory::create<CSRMatrix<VT>>(m0, lowerBound, upperBound);

        auto m2 = DataObjectFactory::create<CSRMatrix<VT>>(rowsTake, numCols, 3, true);
        m2->set(0,0, VT(10.));
        m2->set(0,2, VT(12.));
        m2->set(1,1, VT(21.));

        auto exp = DataObjectFactory::create<CSRMatrix<VT>>(rowsTake, numCols*2, 8, true);
        exp->set(1-lowerBound,1, VT(11.));
        exp->set(2-lowerBound,1, VT(21.));
        exp->set(2-lowerBound,3, VT(23.));

        exp->set(0,numCols, VT(10.));
        exp->set(0,2+numCols, VT(12.));
        exp->set(1,1+numCols, VT(21.));

        colBind<DT, DT, DT>(res, m1, m2, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("View on view") {
        size_t lowerBound = 0;
        size_t upperBound = 2;
        size_t rowsTake = upperBound - lowerBound;
        auto m1 = DataObjectFactory::create<CSRMatrix<VT>>(m0, lowerBound, upperBound);
        auto m2 = DataObjectFactory::create<CSRMatrix<VT>>(m0, upperBound, rowsTake*2);

        auto exp  = DataObjectFactory::create<CSRMatrix<VT>>(rowsTake, numCols*2, numNonZeros, true);
        exp->set(1,1, VT(11.));

        exp->set(2-rowsTake,1+numCols, VT(21.));
        exp->set(2-rowsTake,3+numCols, VT(23.));
        exp->set(3-rowsTake,3+numCols, VT(33.));
        exp->set(3-rowsTake,4+numCols, VT(34.));

        colBind<DT, DT, DT>(res, m1, m2, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1, m2, exp, res);
    }

    DataObjectFactory::destroy(m0);
}