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
#include <runtime/local/kernels/Reverse.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("Reverse", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    auto arg0 = genGivenVals<DT>(3, {
        1, 2,
        3, 4,
        5, 6,
    });
    
    auto exp0 = genGivenVals<DT>(3, {
        5, 6,
        3, 4,
        1, 2,
    });
    
    auto arg1 = genGivenVals<DT>(3, {
       1, 2, 3,
       4, 5, 6,
       7, 8, 9,
    });
    
    auto exp1 = genGivenVals<DT>(3, {
       7, 8, 9,       
       4, 5, 6,
       1, 2, 3,
    });
    
    auto arg2 = genGivenVals<DT>(9, {
       1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9,
    });
    
    auto exp2 = genGivenVals<DT>(9, {
       9,
       8,
       7,
       6,
       5,
       4,
       3,
       2,
       1,
    });
    
    auto arg3 = genGivenVals<DT>(1, {
       1, 2, 3, 4, 5, 6, 7, 8, 9,
    });
    
    auto exp3 = genGivenVals<DT>(1, {
       1, 2, 3, 4, 5, 6, 7, 8, 9,
    });

    DT *res0 = nullptr;
    DT *res1 = nullptr;
    DT *res2 = nullptr;
    DT *res3 = nullptr;

    reverse<DT, DT>(res0, arg0, nullptr);
    reverse<DT, DT>(res1, arg1, nullptr);
    reverse<DT, DT>(res2, arg2, nullptr);
    reverse<DT, DT>(res3, arg3, nullptr);
    CHECK(*res0 == *exp0);
    CHECK(*res1 == *exp1);
    CHECK(*res2 == *exp2);
    CHECK(*res3 == *exp3);
    
    DataObjectFactory::destroy(arg0);
    DataObjectFactory::destroy(arg1);
    DataObjectFactory::destroy(arg2);
    DataObjectFactory::destroy(arg3);
    DataObjectFactory::destroy(exp0);
    DataObjectFactory::destroy(exp1);
    DataObjectFactory::destroy(exp2);
    DataObjectFactory::destroy(exp3);
    DataObjectFactory::destroy(res0);
    DataObjectFactory::destroy(res1);
    DataObjectFactory::destroy(res2);
    DataObjectFactory::destroy(res3);
}

