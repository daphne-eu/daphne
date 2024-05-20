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

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("Reverse", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    DT * arg = nullptr;
    DT * exp = nullptr;
    SECTION("general matrix 1") {
        arg = genGivenVals<DT>(3, {
            1, 2,
            3, 4,
            5, 6,
        });
        exp = genGivenVals<DT>(3, {
            5, 6,
            3, 4,
            1, 2,
        });
    }
    SECTION("general matrix 2") {
        arg = genGivenVals<DT>(3, {
           1, 2, 3,
           4, 5, 6,
           7, 8, 9,
        });
        exp = genGivenVals<DT>(3, {
           7, 8, 9,
           4, 5, 6,
           1, 2, 3,
        });
    }
    SECTION("column matrix") {
        arg = genGivenVals<DT>(9, {
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
        exp = genGivenVals<DT>(9, {
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
    }
    SECTION("row matrix") {
        arg = genGivenVals<DT>(1, {
           1, 2, 3, 4, 5, 6, 7, 8, 9,
        });
        exp = genGivenVals<DT>(1, {
           1, 2, 3, 4, 5, 6, 7, 8, 9,
        });
    }

    DT *res = nullptr;
    reverse<DT, DT>(res, arg, nullptr);
    CHECK(*res == *exp);
    
    DataObjectFactory::destroy(arg, exp, res);
}