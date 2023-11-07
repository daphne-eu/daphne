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
#include <runtime/local/kernels/InsertCol.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("InsertCol", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {0, 1,  2, 3,  4, 5};
    
     
    auto arg = genGivenVals<DT>(3, vals); // 3x2
    SECTION("Check zero-copy") {
        auto ins = genGivenVals<DT>(3, {100, 200, 400}); 
        auto exp = genGivenVals<DT>(3, {100, 1,  200, 3,  400, 5});
        DT * res = nullptr;
        insertCol<DT, DT>(res, arg, ins, 0, 1, nullptr);
        CHECK(*res == *exp);
        // TODO We shouldn't check for pointer equality, since the crucial point
        // about zero-copy here is to not copy the underlying data buffers, while
        // creating a new DenseMatrix object would be okay.
        CHECK(res == arg);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(ins);
        DataObjectFactory::destroy(arg);
    }

    SECTION("NO zero-copy") {
        auto view = DataObjectFactory::create<DT>(arg, 0, 2, 0, 2); 

        auto ins = genGivenVals<DT>(2, {100, 300}); 

        auto expArg = genGivenVals<DT>(3, vals);
        auto expView = genGivenVals<DT>(2, {0, 100, 2, 300});
        DT * res = nullptr;
        insertCol<DT, DT>(res, view, ins, 1, 2, nullptr);
        CHECK(*res == *expView);
        CHECK(*arg == *expArg);
        DataObjectFactory::destroy(expView);
        DataObjectFactory::destroy(expArg);
        DataObjectFactory::destroy(ins);
        DataObjectFactory::destroy(view);
        DataObjectFactory::destroy(res);
        DataObjectFactory::destroy(arg);
    }
}