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
#include <runtime/local/kernels/ColSemiJoin.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

// These test cases are essentially the same as in ColJoinTest.cpp (keep consistent).
// - The only difference is that the colJoin-kernel has the additional result resRhsPos.
// - Note that we assume the rhsData to be unique (primary key), so we don't consider expanding joins here.

#define TEST_NAME "ColSemiJoin"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t
#define STR_VALUE_TYPES std::string

// This is the same as "valid args, string data", just with numeric input data (keep consistent).
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTData *rhsData = nullptr;
    DTPos *resLhsPosExp = nullptr;

    // Empty input data.
    SECTION("empty lhsData, empty rhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = DataObjectFactory::create<DTData>(0, false);
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("empty lhsData, non-empty rhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData, empty rhsData") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        rhsData = DataObjectFactory::create<DTData>(0, false);
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    // Non-empty input data.
    // - The lhsData is assumed to be a foreign key.
    //   - The lhsData could be unique or non-unique.
    //   - The lhsData could be sorted or unsorted.
    // - The rhsData is assumed to be a primary key.
    //   - The rhsData is assumed to be unique.
    //   - The rhsData could be sorted or unsorted.
    // - Matches: the lhsData could contain
    //   - the values in rhsData (all)
    //   - a subset of the values in rhsData (subset)
    //   - a superset of the values in rhsData (superset)
    //   - no values in rhsData (none)

    // (all)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2), VTData(4.4), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2), VTData(4.4), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2), VTData(1.1), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2), VTData(1.1), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // (subset)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(1.1)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(1.1)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    // (superset)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4), VTData(5.5)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4), VTData(5.5)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(5.5), VTData(1.1), VTData(4.4), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(5.5), VTData(1.1), VTData(4.4), VTData(2.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2), VTData(4.4), VTData(4.4), VTData(5.5)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2), VTData(4.4), VTData(4.4), VTData(5.5)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(5.5), VTData(1.1), VTData(4.4), VTData(2.2), VTData(1.1), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3), VTPos(4), VTPos(5)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData(5.5), VTData(1.1), VTData(4.4), VTData(2.2), VTData(1.1), VTData(4.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3), VTPos(4), VTPos(5)});
    }
    // (none)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(22.2), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(22.2), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(44.4), VTData(22.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(44.4), VTData(22.2)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(11.1), VTData(22.2), VTData(44.4), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(11.1), VTData(22.2), VTData(44.4), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(44.4), VTData(222.2), VTData(11.1), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(4.4)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData(11.1), VTData(44.4), VTData(22.2), VTData(11.1), VTData(44.4)});
        rhsData = genGivenVals<DTData>({VTData(1.1), VTData(4.4), VTData(2.2)});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    DTPos *resLhsPosFnd = nullptr;
    colSemiJoin(resLhsPosFnd, lhsData, rhsData, -1, nullptr);
    CHECK(*resLhsPosFnd == *resLhsPosExp);

    DataObjectFactory::destroy(lhsData, rhsData, resLhsPosExp, resLhsPosFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data (keep consistent).
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTData *rhsData = nullptr;
    DTPos *resLhsPosExp = nullptr;

    // Empty input data.
    SECTION("empty lhsData, empty rhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = DataObjectFactory::create<DTData>(0, false);
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("empty lhsData, non-empty rhsData") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData, empty rhsData") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        rhsData = DataObjectFactory::create<DTData>(0, false);
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    // Non-empty input data.
    // - The lhsData is assumed to be a foreign key.
    //   - The lhsData could be unique or non-unique.
    //   - The lhsData could be sorted or unsorted.
    // - The rhsData is assumed to be a primary key.
    //   - The rhsData is assumed to be unique.
    //   - The rhsData could be sorted or unsorted.
    // - Matches: the lhsData could contain
    //   - the values in rhsData (all)
    //   - a subset of the values in rhsData (subset)
    //   - a superset of the values in rhsData (superset)
    //   - no values in rhsData (none)

    // (all)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str4.4"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str4.4"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), all") {
        lhsData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str4.4"), VTData("str2.2"), VTData("str1.1"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), all") {
        lhsData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str4.4"), VTData("str2.2"), VTData("str1.1"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // (subset)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str1.1")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), subset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str1.1")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    // (superset)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4"), VTData("str5.5")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4"), VTData("str5.5")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str5.5"), VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3)});
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str5.5"), VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str4.4"),
                                        VTData("str4.4"), VTData("str5.5")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str4.4"),
                                        VTData("str4.4"), VTData("str5.5")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str5.5"), VTData("str1.1"), VTData("str4.4"), VTData("str2.2"),
                                        VTData("str1.1"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3), VTPos(4), VTPos(5)});
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), superset") {
        lhsData = genGivenVals<DTData>({VTData("str5.5"), VTData("str1.1"), VTData("str4.4"), VTData("str2.2"),
                                        VTData("str1.1"), VTData("str4.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = genGivenVals<DTPos>({VTPos(1), VTPos(2), VTPos(3), VTPos(4), VTPos(5)});
    }
    // (none)
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData("str11.1"), VTData("str22.2"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, sorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData("str11.1"), VTData("str22.2"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>({VTData("str11.1"), VTData("str44.4"), VTData("str22.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (unique, unsorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>({VTData("str11.1"), VTData("str44.4"), VTData("str22.2")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>(
            {VTData("str11.1"), VTData("str11.1"), VTData("str22.2"), VTData("str44.4"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, sorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>(
            {VTData("str11.1"), VTData("str11.1"), VTData("str22.2"), VTData("str44.4"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, sorted), none") {
        lhsData = genGivenVals<DTData>(
            {VTData("str11.1"), VTData("str44.4"), VTData("str222.2"), VTData("str11.1"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str4.4")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("non-empty lhsData (non-unique, unsorted), non-empty rhsData (unique, unsorted), none") {
        lhsData = genGivenVals<DTData>(
            {VTData("str11.1"), VTData("str44.4"), VTData("str22.2"), VTData("str11.1"), VTData("str44.4")});
        rhsData = genGivenVals<DTData>({VTData("str1.1"), VTData("str4.4"), VTData("str2.2")});
        resLhsPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    DTPos *resLhsPosFnd = nullptr;
    colSemiJoin(resLhsPosFnd, lhsData, rhsData, -1, nullptr);
    CHECK(*resLhsPosFnd == *resLhsPosExp);

    DataObjectFactory::destroy(lhsData, rhsData, resLhsPosExp, resLhsPosFnd);
}

// The only possible invalid arg would be a too low numRes. We don't test this case here.