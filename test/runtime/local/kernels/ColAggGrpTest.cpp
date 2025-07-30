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
#include <runtime/local/kernels/ColAggGrp.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "ColAggGrp (" opName ")"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t

template <class DTData, class DTGrpIds, class DTRes>
void checkColAggGrpAndDestroy(AggOpCode opCode, const DTData *data, const DTGrpIds *grpIds, size_t numDistinct,
                              const DTRes *exp, const bool optimisticSplit = false) {
    DTRes *res = nullptr;
    colAggGrp(opCode, res, data, grpIds, numDistinct, optimisticSplit, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(data, grpIds, exp, res);
}

// ****************************************************************************
// Valid arguments
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;
    using VTPos = size_t;
    using DTPos = typename DTData::template WithValueType<VTPos>;

    DTData *data = nullptr;
    DTPos *grpIds = nullptr;
    size_t numDistinct = 0;
    DTData *exp = nullptr;

    // Empty input.
    SECTION("empty input") {
        data = DataObjectFactory::create<DTData>(0, false);
        grpIds = DataObjectFactory::create<DTPos>(0, false);
        numDistinct = 0;
        exp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input.
    // - Given n values, there could be: 1 group, k groups (1 < k < n), or n groups.
    SECTION("non-empty input, 1 group") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 1;
        exp = genGivenVals<DTData>({VTData(6)});
    }
    SECTION("non-empty input, k groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(4), VTData(1), VTData(3), VTData(2)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(2), VTPos(1), VTPos(0), VTPos(1)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(6), VTData(3), VTData(4)});
    }
    SECTION("non-empty input, n groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(0)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(1)});
    }

    checkColAggGrpAndDestroy(AggOpCode::SUM, data, grpIds, numDistinct, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("prod"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;
    using VTPos = size_t;
    using DTPos = typename DTData::template WithValueType<VTPos>;

    DTData *data = nullptr;
    DTPos *grpIds = nullptr;
    size_t numDistinct = 0;
    DTData *exp = nullptr;

    // Empty input.
    SECTION("empty input") {
        data = DataObjectFactory::create<DTData>(0, false);
        grpIds = DataObjectFactory::create<DTPos>(0, false);
        numDistinct = 0;
        exp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input.
    // - Given n values, there could be: 1 group, k groups (1 < k < n), or n groups.
    SECTION("non-empty input, 1 group") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 1;
        exp = genGivenVals<DTData>({VTData(6)});
    }
    SECTION("non-empty input, k groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(4), VTData(1), VTData(3), VTData(2)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(2), VTPos(1), VTPos(0), VTPos(1)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(6), VTData(2), VTData(4)});
    }
    SECTION("non-empty input, n groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(0)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(1)});
    }

    checkColAggGrpAndDestroy(AggOpCode::PROD, data, grpIds, numDistinct, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;
    using VTPos = size_t;
    using DTPos = typename DTData::template WithValueType<VTPos>;

    DTData *data = nullptr;
    DTPos *grpIds = nullptr;
    size_t numDistinct = 0;
    DTData *exp = nullptr;

    // Empty input.
    SECTION("empty input") {
        data = DataObjectFactory::create<DTData>(0, false);
        grpIds = DataObjectFactory::create<DTPos>(0, false);
        numDistinct = 0;
        exp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input.
    // - Given n values, there could be: 1 group, k groups (1 < k < n), or n groups.
    SECTION("non-empty input, 1 group") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 1;
        exp = genGivenVals<DTData>({VTData(1)});
    }
    SECTION("non-empty input, k groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(4), VTData(1), VTData(3), VTData(2)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(2), VTPos(1), VTPos(0), VTPos(1)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(1), VTData(1), VTData(4)});
    }
    SECTION("non-empty input, n groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(0)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(1)});
    }

    checkColAggGrpAndDestroy(AggOpCode::MIN, data, grpIds, numDistinct, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;
    using VTPos = size_t;
    using DTPos = typename DTData::template WithValueType<VTPos>;

    DTData *data = nullptr;
    DTPos *grpIds = nullptr;
    size_t numDistinct = 0;
    DTData *exp = nullptr;

    // Empty input.
    SECTION("empty input") {
        data = DataObjectFactory::create<DTData>(0, false);
        grpIds = DataObjectFactory::create<DTPos>(0, false);
        numDistinct = 0;
        exp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input.
    // - Given n values, there could be: 1 group, k groups (1 < k < n), or n groups.
    SECTION("non-empty input, 1 group") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 1;
        exp = genGivenVals<DTData>({VTData(3)});
    }
    SECTION("non-empty input, k groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(4), VTData(1), VTData(3), VTData(2)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(2), VTPos(1), VTPos(0), VTPos(1)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(4)});
    }
    SECTION("non-empty input, n groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(0)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(1)});
    }

    checkColAggGrpAndDestroy(AggOpCode::MAX, data, grpIds, numDistinct, exp);
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
    using DTData = TestType;
    using DTPos = typename DTData::template WithValueType<size_t>;

    auto data = genGivenVals<DTData>({1});
    auto grpIds = genGivenVals<DTPos>({0});
    size_t numDistinct = 1;

    DTData *res = nullptr;
    CHECK_THROWS(colAggGrp(static_cast<AggOpCode>(999), res, data, grpIds, numDistinct, false, nullptr));

    DataObjectFactory::destroy(data);
    DataObjectFactory::destroy(grpIds);
    if (res)
        DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("any") ": size mismatch", TAG_KERNELS, (DATA_TYPES), (double)) {
    using DTData = TestType;
    using DTPos = typename DTData::template WithValueType<size_t>;

    auto data = genGivenVals<DTData>({1, 2, 3});
    auto grpIds = genGivenVals<DTPos>({0, 1});
    size_t numDistinct = 2;

    DTData *res = nullptr;
    CHECK_THROWS(colAggGrp(AggOpCode::SUM, res, data, grpIds, numDistinct, false, nullptr));

    DataObjectFactory::destroy(data, grpIds);
    if (res)
        DataObjectFactory::destroy(res);
}

// ****************************************************************************
// With Optimistic Spliting
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum") ": Optimistic Spliting", TAG_KERNELS, (DATA_TYPES),
                           (int64_t, uint64_t, int32_t, uint32_t)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;
    using HalfTypeT = typename HalfType<VTData>::type;
    using VTPos = size_t;
    using DTPos = typename DTData::template WithValueType<VTPos>;

    DTData *data = nullptr;
    DTPos *grpIds = nullptr;
    size_t numDistinct = 0;
    DTData *exp = nullptr;

    SECTION("empty input") {
        data = DataObjectFactory::create<DTData>(0, false);
        grpIds = DataObjectFactory::create<DTPos>(0, false);
        numDistinct = 0;
        exp = DataObjectFactory::create<DTData>(0, false);
    }

    SECTION("non-empty input, 1 group") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 1;
        exp = genGivenVals<DTData>({VTData(6)});
    }
    SECTION("non-empty input, k groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(4), VTData(1), VTData(3), VTData(2)});
        grpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(2), VTPos(1), VTPos(0), VTPos(1)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(6), VTData(3), VTData(4)});
    }
    SECTION("non-empty input, n groups") {
        data = genGivenVals<DTData>({VTData(2), VTData(1), VTData(3)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(0)});
        numDistinct = 3;
        exp = genGivenVals<DTData>({VTData(3), VTData(2), VTData(1)});
    }

    SECTION("non-empty input, k groups, boundary values") {
        // Set boundary value based on the value type
        VTData valueMax = std::numeric_limits<HalfTypeT>::max();
        VTData valueMin;

        if constexpr (std::is_same_v<VTData, uint64_t> || std::is_same_v<VTData, uint32_t>) {
            valueMin = HalfTypeT(0);
        } else {
            valueMin = std::numeric_limits<HalfTypeT>::min();
        }

        data = genGivenVals<DTData>(
            std::vector<VTData>{VTData(valueMax), VTData(10), VTData(valueMin), VTData(-10), VTData(5)});
        grpIds = genGivenVals<DTPos>({VTPos(1), VTPos(1), VTPos(0), VTPos(0), VTPos(0)});
        numDistinct = 2;
        VTData expVal0 = VTData(valueMin) + VTData(-5);
        VTData expVal1 = VTData(valueMax) + VTData(10);
        exp = genGivenVals<DTData>({expVal0, expVal1});
    }

    checkColAggGrpAndDestroy(AggOpCode::SUM, data, grpIds, numDistinct, exp, true);
}
