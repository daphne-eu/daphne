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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColSelectCmp.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#define TEST_NAME(opName) "ColSelectCmp (" opName ")"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t
#define STR_VALUE_TYPES std::string

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(2.2);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = genGivenVals<DTPos>({0, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = genGivenVals<DTPos>({0, 1, 2});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::EQ, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str2.2";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str1.1";
        resPosExp = genGivenVals<DTPos>({0, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str1.1", "str1.1"});
        rhsData = "str1.1";
        resPosExp = genGivenVals<DTPos>({0, 1, 2});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::EQ, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES),
                           (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(2.2);
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::NEQ, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str1.1", "str1.1"});
        rhsData = "str1.1";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str1.1";
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str2.2";
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::NEQ, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(6.6);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(2.2);
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(0.0);
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::GT, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str6.6";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str2.2";
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str0.0";
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::GT, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(6.6);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(3.3);
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::GE, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str6.6";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str3.3";
        resPosExp = genGivenVals<DTPos>({1, 2});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str1.1";
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::GE, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(1.1);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(3.3);
        resPosExp = genGivenVals<DTPos>({0, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(6.6);
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::LT, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str1.1";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str3.3";
        resPosExp = genGivenVals<DTPos>({0, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str6.6";
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::LT, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le") ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = VTData(123);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(0.0);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(3.3);
        resPosExp = genGivenVals<DTPos>({0, 2, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(5.5), VTData(3.3), VTData(1.1)});
        rhsData = VTData(5.5);
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::LE, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le") ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    VTData rhsData;
    DTPos *resPosExp = nullptr;

    // Empty input data -> empty result positions.
    SECTION("empty lhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = "str123";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data.
    // - "no", "some", or "all" of the lhs input values could match the rhs input value.
    SECTION("non-empty lhsData (no matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str0.0";
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (some matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str3.3";
        resPosExp = genGivenVals<DTPos>({0, 2, 3});
    }
    SECTION("non-empty lhsData (all matches)") {
        lhsData = genGivenVals<DTData>({"str1.1", "str5.5", "str3.3", "str1.1"});
        rhsData = "str5.5";
        resPosExp = genGivenVals<DTPos>({0, 1, 2, 3});
    }

    DTPos *resPosFnd = nullptr;
    colSelectCmp(CmpOpCode::LE, resPosFnd, lhsData, rhsData, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsData, resPosExp, resPosFnd);
}