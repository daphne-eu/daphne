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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TEST_NAME(opName) "AggAll (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkAggAll(AggOpCode opCode, const DT * arg, typename DT::VT exp) {
    typename DT::VT res = aggAll<DT>(opCode, arg, nullptr);
    CHECK(res == exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        3, 0, 2, 0,
        0, 0, 1, 1,
        2, 5, 0, 0,
    });
    
    checkAggAll(AggOpCode::SUM, m0, 0);
    checkAggAll(AggOpCode::SUM, m1, 14);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m2 = genGivenVals<DT>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    
    checkAggAll(AggOpCode::MIN, m0, 0);
    checkAggAll(AggOpCode::MIN, m1, 2);
    checkAggAll(AggOpCode::MIN, m2, 0);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        5, 2, 8, 9,
        7, 4, 5, 4,
    });
    auto m2 = genGivenVals<DT>(3, {
        4, 0, 0, 9,
        0, 2, 0, 0,
        0, 0, 5, 0,
    });
    
    checkAggAll(AggOpCode::MAX, m0, 0);
    checkAggAll(AggOpCode::MAX, m1, 9);
    checkAggAll(AggOpCode::MAX, m2, 9);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}


TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(3, {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(3, {
        4, 6, 3, 9,
        2, 2, 8, 9,
        4, 4, 5, 4,
    });
    auto m2 = genGivenVals<DT>(3, {
        4, 0, 0, 9,
        0, 6, 0, 0,
        0, 0, 5, 0,
    });
    
    checkAggAll(AggOpCode::MEAN, m0, 0);
    checkAggAll(AggOpCode::MEAN, m1, 5);
    checkAggAll(AggOpCode::MEAN, m2, 2);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}