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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColBind.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("ColBind", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
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
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
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
    
    Frame * f234;
    
    Frame * res = nullptr;
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
    }
    SECTION("non-unique labels") {
        std::vector<Structure *> cols1 = {c2, c3, c4};
        std::string labels1[] = {l0, l1, "c"};
        f234 = DataObjectFactory::create<Frame>(cols1, labels1);
        CHECK_THROWS(colBind<Frame, Frame, Frame>(res, f01, f234, nullptr));
    }
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(c3);
    DataObjectFactory::destroy(c4);
    DataObjectFactory::destroy(f01);
    DataObjectFactory::destroy(f234);
    DataObjectFactory::destroy(res);
}