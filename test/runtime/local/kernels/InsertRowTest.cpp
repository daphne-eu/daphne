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
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/InsertRow.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("InsertRow", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
     0, 1, 
     2, 3,  
     4, 5};
    
    
    SECTION("Check zero-copy") {
        auto arg = genGivenVals<DT>(3, vals); 

        auto ins = genGivenVals<DT>(1, {8, 9}); 
        auto exp = genGivenVals<DT>(3, {0, 1,  8, 9,  4, 5});
        DT * res = nullptr;
        insertRow<DT, DT>(res, arg, ins, 1, 2, nullptr);
        CHECK(*res == *exp);
        CHECK(res == arg);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(ins);
        DataObjectFactory::destroy(arg);
    }

    SECTION("NO zero-copy") {
        auto arg = genGivenVals<DT>(3, vals);
        auto view = DataObjectFactory::create<DT>(arg, 0, 2, 0, 2); 

        auto ins = genGivenVals<DT>(1, {8, 9}); 

        auto exp_arg = genGivenVals<DT>(3, vals);
        auto exp_view = genGivenVals<DT>(2, {0, 1,  8, 9});
        DT * res = nullptr;
        insertRow<DT, DT>(res, view, ins, 1, 2, nullptr);
        CHECK(*res == *exp_view);
        CHECK(*arg == *exp_arg);
        DataObjectFactory::destroy(exp_view);
        DataObjectFactory::destroy(exp_arg);
        DataObjectFactory::destroy(ins);
        DataObjectFactory::destroy(view);
        DataObjectFactory::destroy(res);
        DataObjectFactory::destroy(arg);
    }
}