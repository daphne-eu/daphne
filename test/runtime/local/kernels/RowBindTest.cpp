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
#include <runtime/local/kernels/RowBind.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("RowBind", TAG_KERNELS, (DenseMatrix, Matrix), (double, uint32_t)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
    
    DT * res = nullptr;
    SECTION("matching matrices") {
        auto m1 = genGivenVals<DT>(3, {
           17, 20, 30, 40,
           50, 60, 70, 80,
           90, 100, 110, 120,
        });

        auto exp = genGivenVals<DT>(7, {
           1, 2, 3, 4,
           5, 6, 7, 8, 
           9, 10, 11, 12,
           13, 14, 15, 16,
           17, 20, 30, 40, 
           50, 60, 70, 80,
           90, 100, 110, 120
        });

        rowBind<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1, exp, res);
    }
    SECTION("size mismatch") {
        auto m1 = genGivenVals<DT>(3, {
           17, 20, 30,
           50, 60, 70,
           90, 100, 110,
        });

        CHECK_THROWS(rowBind<DT, DT, DT>(res, m0, m1, nullptr));
        
        DataObjectFactory::destroy(m1);
    }
    
    DataObjectFactory::destroy(m0);
}

TEMPLATE_TEST_CASE("RowBind", TAG_KERNELS, (Frame)) {
    auto c0 = genGivenVals<DenseMatrix<double>>(4, {1, 5, 9, 13});
    auto c1 = genGivenVals<DenseMatrix<float>>(4, {2, 6, 10, 14});
    auto c2 = genGivenVals<DenseMatrix<int64_t>>(4, {3, 7, 11, 15});
    auto c3 = genGivenVals<DenseMatrix<uint8_t>>(4, {4, 8, 12, 16});
    auto c4 = genGivenVals<DenseMatrix<double>>(3, {17, 50, 90});
    auto c5 = genGivenVals<DenseMatrix<float>>(3, {20, 60, 70});
    auto c6 = genGivenVals<DenseMatrix<int64_t>>(3, {30, 70, 110});
    auto c7 = genGivenVals<DenseMatrix<uint8_t>>(3, {40, 80, 120});
    auto c04 = genGivenVals<DenseMatrix<double>>(7, {1, 5, 9, 13, 17, 50, 90});
    auto c15 = genGivenVals<DenseMatrix<float>>(7, {2, 6, 10, 14, 20, 60, 70});
    auto c26 = genGivenVals<DenseMatrix<int64_t>>(7, {3, 7, 11, 15, 30, 70, 110});
    auto c37 = genGivenVals<DenseMatrix<uint8_t>>(7, {4, 8, 12, 16, 40, 80, 120});

    std::string l0 = "a";
    std::string l1 = "b";
    std::string l2 = "c";
    std::string l3 = "d";
    
    
    std::vector<Structure *> cols0123 = {c0, c1, c2, c3};
    std::string labels0123[] = {l0, l1, l2, l3};
    auto f0123 = DataObjectFactory::create<Frame>(cols0123, labels0123);
    
    Frame * res = nullptr;
    SECTION("matching frames") {
        std::vector<Structure *> cols4567 = {c4, c5, c6, c7};
        Frame * f4567 = DataObjectFactory::create<Frame>(cols4567, labels0123);
        
        rowBind<Frame, Frame, Frame>(res, f0123, f4567, nullptr);

        // Check dimensions.
        REQUIRE(res->getNumRows() == 7);
        REQUIRE(res->getNumCols() == 4);

        // Check column types.
        const std::string * labelsRes = res->getLabels();
        CHECK(labelsRes[0] == l0);
        CHECK(labelsRes[1] == l1);
        CHECK(labelsRes[2] == l2);
        CHECK(labelsRes[3] == l3);

        // Check row labels.
        const ValueTypeCode * schemaRes = res->getSchema();
        CHECK(schemaRes[0] == ValueTypeCode::F64);
        CHECK(schemaRes[1] == ValueTypeCode::F32);
        CHECK(schemaRes[2] == ValueTypeCode::SI64);
        CHECK(schemaRes[3] == ValueTypeCode::UI8);

        // Check column data. 
        CHECK(*(res->getColumn<double>(0)) == *c04);
        CHECK(*(res->getColumn<float>(1)) == *c15);
        CHECK(*(res->getColumn<int64_t>(2)) == *c26);
        CHECK(*(res->getColumn<uint8_t>(3)) == *c37);
        
        DataObjectFactory::destroy(f4567);
        DataObjectFactory::destroy(res);
    }
    SECTION("size mismatch") {
        std::string labels012[] = {l0, l1, l2};
        std::vector<Structure *> cols456 = {c4, c5, c6};
        Frame * f456 = DataObjectFactory::create<Frame>(cols456, labels012);
        
        CHECK_THROWS(rowBind<Frame, Frame, Frame>(res, f0123, f456, nullptr));
    }
    SECTION("schema mismatch") {
        std::vector<Structure *> cols4444 = {c4, c4, c4, c4};
        Frame * f4444 = DataObjectFactory::create<Frame>(cols4444, labels0123);
        
        CHECK_THROWS(rowBind<Frame, Frame, Frame>(res, f0123, f4444, nullptr));
    }
    SECTION("label mismatch") {
        std::string labels3210[] = {l3, l2, l1, l0};
        std::vector<Structure *> cols4567 = {c4, c5, c6, c7};
        Frame * f4567 = DataObjectFactory::create<Frame>(cols4567, labels3210);
        
        CHECK_THROWS(rowBind<Frame, Frame, Frame>(res, f0123, f4567, nullptr));
    }
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(c3);
    DataObjectFactory::destroy(c4);
    DataObjectFactory::destroy(c5);
    DataObjectFactory::destroy(c6);
    DataObjectFactory::destroy(c7);
    DataObjectFactory::destroy(c04);
    DataObjectFactory::destroy(c15);
    DataObjectFactory::destroy(c26);
    DataObjectFactory::destroy(c37);
    DataObjectFactory::destroy(f0123);
}

TEMPLATE_PRODUCT_TEST_CASE("RowBind", TAG_KERNELS, (CSRMatrix), (double, uint32_t)) {
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

        auto exp = DataObjectFactory::create<CSRMatrix<VT>>(numRows*2, numCols, 8, true);
        exp->set(1,1, VT(11.));
        exp->set(2,1, VT(21.));
        exp->set(2,3, VT(23.));
        exp->set(3,3, VT(33.));
        exp->set(3,4, VT(34.));
        exp->set(1 + numRows,0, VT(10.));
        exp->set(1 + numRows,2, VT(12.));
        exp->set(3 + numRows,0, VT(30.));

        rowBind<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("Normal on view") {
        size_t lower_bound = 1;
        size_t upper_bound = 3;
        auto m1 = DataObjectFactory::create<CSRMatrix<VT>>(m0, lower_bound, upper_bound);
        auto exp = DataObjectFactory::create<CSRMatrix<VT>>(numRows + (upper_bound - lower_bound), numCols, 8, true);
        exp->set(1,1, VT(11.));
        exp->set(2,1, VT(21.));
        exp->set(2,3, VT(23.));
        exp->set(3,3, VT(33.));
        exp->set(3,4, VT(34.));
        exp->set(1 + (numRows-1), 1, VT(11.));
        exp->set(2 + (numRows-1), 1, VT(21.));
        exp->set(2 + (numRows-1), 3, VT(23.));

        rowBind<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("View on view") {
        auto m1 = DataObjectFactory::create<CSRMatrix<VT>>(m0, 2, 4);
        auto m2 = DataObjectFactory::create<CSRMatrix<VT>>(m0, 0, 2);
        auto exp  = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, true);
        exp->set(0,1, VT(21.));
        exp->set(0,3, VT(23.));
        exp->set(1,3, VT(33.));
        exp->set(1,4, VT(34.));
        exp->set(3,1, VT(11.));

        rowBind<DT, DT, DT>(res, m1, m2, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(m2);
        DataObjectFactory::destroy(res);
    }

    DataObjectFactory::destroy(m0);
}