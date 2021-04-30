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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/AggRow.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggRow (" opName ")"
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkAggRow(AggOpCode opCode, const DT * arg, const DT * exp) {
    DT * res = nullptr;
    aggRow<DT>(opCode, res, arg);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DT>(3, {0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        3, 0, 2, 0,
        0, 0, 1, 1,
        2, 5, 0, 0,
    });
    auto m1exp = genGivenVals<DT>(3, {5, 2, 7});
    
    checkAggRow(AggOpCode::SUM, m0, m0exp);
    checkAggRow(AggOpCode::SUM, m1, m1exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DT>(3, {0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DT>(3, {3, 2, 4});
    
    checkAggRow(AggOpCode::MIN, m0, m0exp);
    checkAggRow(AggOpCode::MIN, m1, m1exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DT>(3, {0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DT>(3, {9, 9, 7});
    
    checkAggRow(AggOpCode::MAX, m0, m0exp);
    checkAggRow(AggOpCode::MAX, m1, m1exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
}