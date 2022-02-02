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
#include <runtime/local/kernels/RowBind.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("RowBind", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
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
    
    DT * res = nullptr;
    ValueTypeCode schemas[] = {};
    std::string labels0123[] = {};
        
    rowBind<DT, DT, DT>(res, m0, m1, schemas, labels0123, nullptr);
    CHECK(*res == *exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}

TEST_CASE("RowBind - Frame", TAG_KERNELS) {
    auto c0 = genGivenVals<DenseMatrix<double>>(4, {1, 5, 9, 13});
    auto c1 = genGivenVals<DenseMatrix<double>>(4, {2, 6, 10, 14});
    auto c2 = genGivenVals<DenseMatrix<double>>(4, {3, 7, 11, 15});
    auto c3 = genGivenVals<DenseMatrix<double>>(4, {4, 8, 12, 16});
    auto c4 = genGivenVals<DenseMatrix<double>>(3, {17, 50, 90});
    auto c5 = genGivenVals<DenseMatrix<double>>(3, {20, 60, 70});
    auto c6 = genGivenVals<DenseMatrix<double>>(3, {30, 70, 110});
    auto c7 = genGivenVals<DenseMatrix<double>>(3, {40, 80, 120});
//     auto c04 = genGivenVals<DenseMatrix<double>>(7, {1, 5, 9, 13, 17, 50, 90});
//     auto c15 = genGivenVals<DenseMatrix<double>>(7, {2, 6, 10, 14, 20, 60, 70});
//     auto c26 = genGivenVals<DenseMatrix<double>>(7, {3, 7, 11, 15, 30, 70, 110});
//     auto c37 = genGivenVals<DenseMatrix<double>>(7, {4, 8, 12, 16, 40, 80, 120});

    std::string l0 = "a";
    std::string l1 = "b";
    std::string l2 = "c";
    std::string l3 = "d";
    
    
    std::vector<Structure *> cols0123 = {c0, c1, c2, c3};
    ValueTypeCode schemas[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64}; 
    std::string labels0123[] = {l0, l1, l2, l3};
    auto f0123 = DataObjectFactory::create<Frame>(cols0123, labels0123);
    
    Frame * res = nullptr;
    Frame * f4567;
    SECTION("unique labels") {
        std::vector<Structure *> cols4567 = {c4, c5, c6, c7};
        f4567 = DataObjectFactory::create<Frame>(cols4567, labels0123);
        rowBind<Frame, Frame, Frame>(res, f0123, f4567, schemas, labels0123, nullptr);
        
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
        CHECK(schemaRes[1] == ValueTypeCode::F64);
        CHECK(schemaRes[2] == ValueTypeCode::F64);
        CHECK(schemaRes[3] == ValueTypeCode::F64);

        // Check column data. 
//         CHECK(*(res->getColumn<double>(0)) == *c04);
//         CHECK(*(res->getColumn<double>(1)) == *c15);
//         CHECK(*(res->getColumn<double>(2)) == *c26);
//         CHECK(*(res->getColumn<double>(3)) == *c37);

    }
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(c3);
    DataObjectFactory::destroy(c4);
    DataObjectFactory::destroy(c5);
    DataObjectFactory::destroy(c6);
    DataObjectFactory::destroy(c7);
    DataObjectFactory::destroy(f0123);
    DataObjectFactory::destroy(f4567);
    DataObjectFactory::destroy(res);
}
