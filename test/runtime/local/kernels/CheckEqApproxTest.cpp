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

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, original matrices", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double)) {
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
        CHECK(checkEqApprox(m1, m1,0.00001, nullptr)); 
    }
    SECTION("diff inst, same size, same cont") {
        auto m2 = genGivenVals<DT>(4, vals);
        CHECK(checkEqApprox(m1, m2,0.00001, nullptr));
    }
    SECTION("diff inst, diff size, same cont") {
        auto m2 = genGivenVals<DT>(6, vals);
        CHECK_FALSE(checkEqApprox(m1, m2,0.00001, nullptr));
    }
    SECTION("diff inst, same size, accepted difference default ESP") {
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK(checkEqApprox(m1, m2,0.00001, nullptr));
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
   


TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, views on matrices", TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        1, 2, 2, 2, 0, 0,
        3, 4, 4, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };
    std::vector<typename DT::VT> vals2 = { 
        1.001, 2, 2, 2, 0, 0,
        3, 4, 4.001, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };    

    auto orig1 = genGivenVals<DT>(6, vals);
    auto orig2 = genGivenVals<DT>(6, vals2); 
    
    SECTION("same inst") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        CHECK(checkEqApprox(view1, view1,0.00001, nullptr));
    }
    SECTION("same view on different equal matrices") {
        auto view1 = DataObjectFactory::create<DT>(orig1, 0, 2, 0, 2);
        auto view2 = DataObjectFactory::create<DT>(orig2, 0, 2, 0, 2);
        CHECK(checkEqApprox(view1, view2,0.01, nullptr));
        CHECK_FALSE(checkEqApprox(view1, view2,0.000000001, nullptr));
    }
}
