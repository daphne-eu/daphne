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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>
#include <runtime/local/kernels/CheckEqApprox.h>

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, original matrices", TAG_KERNELS, (CSRMatrix), (double)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 1, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6, 7,
        0, 8, 0, 0, 9, 0,
    };
    std::vector<typename DT::VT> vals2 = { 
        0, 0, 1.0000001, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6.0000001, 7,
        0, 8, 0, 0, 9, 0,
    };
    auto m1 = genGivenVals<DT>(4, vals);
    SECTION("same inst") {
        CHECK(*m1 == *m1);
    }
    SECTION("diff inst, same size, same cont") {
        auto m2 = genGivenVals<DT>(4, vals);
        CHECK(*m1 == *m2);
    }
    SECTION("diff inst, diff size, same cont") {
        auto m2 = genGivenVals<DT>(6, vals);
        CHECK_FALSE(*m1 == *m2);
    }
    SECTION("diff inst, same size, accepted difference default ESP") {
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK(*m1 == *m2);
    }
    SECTION("diff inst, same size, accepted difference defined ESP") {
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK(checkEqApprox<DT>(m1, m2,0.01, nullptr));
    }
    SECTION("diff inst, same size, unaccepted difference defined ESP"){
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK_FALSE(checkEqApprox<DT>(m1, m2,0.0000000000001, nullptr)); 
    }
}
/*    
TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, views on matrices", TAG_KERNELS, (DenseMatrix), (doubl)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        1, 2, 2, 2, 0, 0,
        3, 4, 4, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };
    auto orig1 = genGivenVals<DT>(6, vals);
    
    SECTION("same inst") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        CHECK(*view1 == *view1);
    }
    SECTION("diff inst, same size, same cont, same orig") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view3 = DataObjectFactory::create<DT>(orig1, 1, 3, 4, 6);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
    }
    SECTION("diff inst, same size, same cont, diff orig") {
        auto orig2 = genGivenVals<DT>(6, vals);
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig2, 0, 2, 0, 2);
        auto view3 = DataObjectFactory::create<DT>(orig2, 1, 3, 4, 6);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
    }
    SECTION("diff inst, same size, same cont, overlap") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 1, 3);
        auto view2 = DataObjectFactory::create<DT>(orig1, 0, 2, 2, 4);
        CHECK(*view1 == *view2);
    }
    SECTION("diff inst, same size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig1, 4, 6, 4, 6);
        CHECK_FALSE(*view1 == *view2);
    }
    SECTION("diff inst, diff size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig1, 0, 3, 0, 2);
        CHECK_FALSE(*view1 == *view2);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, views on matrices", TAG_KERNELS, (CSRMatrix), (double)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 0, 0,
        0, 1, 0, 2,
        3, 0, 0, 0,
        0, 0, 4, 5,
        0, 0, 0, 0,
        3, 0, 0, 0,
        0, 0, 4, 5,
        0, 0, 4, 5,
        0, 0, 4, 5,
    };
    auto orig1 = genGivenVals<DT>(9, vals);
    
    SECTION("same inst") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 1, 4);
        CHECK(*view1 == *view1);
    }
    SECTION("diff inst, same size, same cont, same orig") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view2 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view3 = DataObjectFactory::create<DT>(orig1, 5, 7);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
    }
    SECTION("diff inst, same size, same cont, diff orig") {
        auto orig2 = genGivenVals<DT>(9, vals);
        auto view1 = DataObjectFactory::create<DT>(orig1, 2, 4);
        auto view2 = DataObjectFactory::create<DT>(orig2, 2, 4);
        auto view3 = DataObjectFactory::create<DT>(orig2, 5, 7);
        CHECK(*view1 == *view2);
        CHECK(*view1 == *view3);
    }
    SECTION("diff inst, same size, same cont, overlap") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 6, 8);
        auto view2 = DataObjectFactory::create<DT>(orig1, 7, 9);
        CHECK(*view1 == *view2);
    }
    SECTION("diff inst, same size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 3);
        auto view2 = DataObjectFactory::create<DT>(orig1, 3, 6);
        CHECK_FALSE(*view1 == *view2);
    }
    SECTION("diff inst, diff size, diff cont") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 3);
        auto view2 = DataObjectFactory::create<DT>(orig1, 3, 7);
        CHECK_FALSE(*view1 == *view2);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, empty matrices", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    auto orig1 = genGivenVals<DT>(3, vals);
    
    SECTION("orig, diff inst, same size") {
        auto orig2 = genGivenVals<DT>(3, vals);
        CHECK(*orig1 == *orig2);
    }
    SECTION("view, diff inst, same size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 4);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3, 0, 4);
        CHECK(*view1 == *view2);
    }
    SECTION("view, diff inst, diff size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 1, 0, 4);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3, 0, 4);
        CHECK_FALSE(*view1 == *view2);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEq, empty matrices", TAG_KERNELS, (CSRMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    auto orig1 = genGivenVals<DT>(3, vals);
    
    SECTION("orig, diff inst, same size") {
        auto orig2 = genGivenVals<DT>(3, vals);
        CHECK(*orig1 == *orig2);
    }
    SECTION("view, diff inst, same size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3);
        CHECK(*view1 == *view2);
    }
    SECTION("view, diff inst, diff size") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 1);
        auto view2 = DataObjectFactory::create<DT>(orig1, 1, 3);
        CHECK_FALSE(*view1 == *view2);
    }
}*/
