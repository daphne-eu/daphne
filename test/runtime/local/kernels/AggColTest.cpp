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
#include <runtime/local/kernels/AggCol.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggCol (" opName ")"
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkAggCol(AggOpCode opCode, const DT * arg, const DT * exp) {
    DT * res = nullptr;
    aggCol<DT>(opCode, res, arg);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m0exp = genGivenVals<DT>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        3, 0, 2, 0,
        0, 0, 1, 1,
        2, 5, 0, 0,
    });
    auto m1exp = genGivenVals<DT>(1, {5, 5, 3, 1});
    
    checkAggCol(AggOpCode::SUM, m0, m0exp);
    checkAggCol(AggOpCode::SUM, m1, m1exp);
    
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
    auto m0exp = genGivenVals<DT>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DT>(1, {4, 2, 3, 4});
    
    checkAggCol(AggOpCode::MIN, m0, m0exp);
    checkAggCol(AggOpCode::MIN, m1, m1exp);
    
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
    auto m0exp = genGivenVals<DT>(1, {0, 0, 0, 0});
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m1exp = genGivenVals<DT>(1, {7, 6, 8, 9});
    
    checkAggCol(AggOpCode::MAX, m0, m0exp);
    checkAggCol(AggOpCode::MAX, m1, m1exp);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m0exp);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m1exp);
}