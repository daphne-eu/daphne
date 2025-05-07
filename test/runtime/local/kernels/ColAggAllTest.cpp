/*
 * Copyright 2025 The DAPHNE Consortium
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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColAggAll.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "ColAggAll (" opName ")"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t

template <class DTArg, class DTRes>
void checkColAggAllAndDestroy(AggOpCode opCode, const DTArg *arg, const DTRes *exp) {
    DTRes *res = nullptr;
    colAggAll(opCode, res, arg, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(arg, exp, res);
}

// ****************************************************************************
// Valid arguments
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    AggOpCode opCode = AggOpCode::SUM;
    DT *arg = nullptr;
    DT *exp = nullptr;

    SECTION("empty input") {
        arg = DataObjectFactory::create<DT>(0, false);
        exp = genGivenVals<DT>({AggOpCodeUtils::getNeutral<VT>(opCode)});
    }
    SECTION("non-empty input") {
        arg = genGivenVals<DT>({VT(2), VT(1), VT(3)});
        exp = genGivenVals<DT>({VT(6)});
    }

    checkColAggAllAndDestroy(opCode, arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("prod"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    AggOpCode opCode = AggOpCode::PROD;
    DT *arg = nullptr;
    DT *exp = nullptr;

    SECTION("empty input") {
        arg = DataObjectFactory::create<DT>(0, false);
        exp = genGivenVals<DT>({AggOpCodeUtils::getNeutral<VT>(opCode)});
    }
    SECTION("non-empty input, without zero") {
        arg = genGivenVals<DT>({VT(2), VT(1), VT(0)});
        exp = genGivenVals<DT>({VT(0)});
    }
    SECTION("non-empty input, with zero") {
        arg = genGivenVals<DT>({VT(2), VT(1), VT(3)});
        exp = genGivenVals<DT>({VT(6)});
    }

    checkColAggAllAndDestroy(opCode, arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    AggOpCode opCode = AggOpCode::MIN;
    DT *arg = nullptr;
    DT *exp = nullptr;

    SECTION("empty input") {
        arg = DataObjectFactory::create<DT>(0, false);
        exp = genGivenVals<DT>({AggOpCodeUtils::getNeutral<VT>(opCode)});
    }
    SECTION("non-empty input") {
        arg = genGivenVals<DT>({VT(2), VT(1), VT(3), VT(2)});
        exp = genGivenVals<DT>({VT(1)});
    }

    checkColAggAllAndDestroy(opCode, arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    AggOpCode opCode = AggOpCode::MAX;
    DT *arg = nullptr;
    DT *exp = nullptr;

    SECTION("empty input") {
        arg = DataObjectFactory::create<DT>(0, false);
        exp = genGivenVals<DT>({AggOpCodeUtils::getNeutral<VT>(opCode)});
    }
    SECTION("non-empty input") {
        arg = genGivenVals<DT>({VT(2), VT(1), VT(3), VT(2)});
        exp = genGivenVals<DT>({VT(3)});
    }

    checkColAggAllAndDestroy(opCode, arg, exp);
}

// TODO IDXMIX
// TODO IDXMAX
// TODO MEAN
// TODO STDDEV
// TODO VAR

// ****************************************************************************
// Invalid arguments
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>({1});

    DT *res = nullptr;
    CHECK_THROWS(colAggAll(static_cast<AggOpCode>(999), res, arg, nullptr));

    DataObjectFactory::destroy(arg);
    if (res)
        DataObjectFactory::destroy(res);
}