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
#include <runtime/local/kernels/ColGroupFirst.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#define TEST_NAME "ColGroupFirst"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t
#define STR_VALUE_TYPES std::string

// This is the same as "valid args, string data", just with numeric input data (keep consistent).
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *argData = nullptr;
    DTPos *resGrpIdsExp = nullptr;
    DTPos *resReprPosExp = nullptr;

    // Empty input data.
    SECTION("empty argData") {
        argData = DataObjectFactory::create<DTData>(0, false);
        resGrpIdsExp = DataObjectFactory::create<DTPos>(0, false);
        resReprPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data, one distinct value.
    SECTION("non-empty argData (one distinct value)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0)});
    }
    // Non-empty input data, unique values.
    // - The input values could be sorted or unsorted.
    SECTION("non-empty argData (unique values, sorted)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(3.3), VTData(4.4), VTData(5.5)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values, unsorted)") {
        argData = genGivenVals<DTData>({VTData(3.3), VTData(1.1), VTData(2.2), VTData(5.5), VTData(4.4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // Non-empty input data, a few distinct values.
    // - The input values could be
    //   - sorted
    //   - unsorted and clustered (all occurrences of each distinct value in a contiguous subsequence)
    //   - unsorted and unclustered (occurrences of a distict value can be separated from each other)
    SECTION("non-empty argData (a few distinct values, sorted)") {
        argData = genGivenVals<DTData>(
            {VTData(1.1), VTData(1.1), VTData(2.2), VTData(3.3), VTData(3.3), VTData(3.3), VTData(4.4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(1), VTPos(2), VTPos(2), VTPos(2), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(2), VTPos(3), VTPos(6)});
    }
    SECTION("non-empty argData (a few distinct values, unsorted+clustered)") {
        argData = genGivenVals<DTData>(
            {VTData(2.2), VTData(1.1), VTData(1.1), VTData(4.4), VTData(3.3), VTData(3.3), VTData(3.3)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(1), VTPos(2), VTPos(3), VTPos(3), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (a few distinct values, unsorted+unclustered)") {
        argData = genGivenVals<DTData>(
            {VTData(2.2), VTData(3.3), VTData(1.1), VTData(4.4), VTData(3.3), VTData(3.3), VTData(1.1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(1), VTPos(1), VTPos(2)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3)});
    }

    DTPos *resGrpIdsFnd = nullptr;
    DTPos *resReprPosFnd = nullptr;
    colGroupFirst(resGrpIdsFnd, resReprPosFnd, argData, nullptr);
    CHECK(*resGrpIdsFnd == *resGrpIdsExp);
    CHECK(*resReprPosFnd == *resReprPosExp);

    DataObjectFactory::destroy(argData, resGrpIdsFnd, resReprPosFnd, resGrpIdsExp, resReprPosExp);
}

// This is the same as "valid args, numeric data", just with string-valued input data (keep consistent).
// - Used the following regex replace: `VTData\((\d+\.\d+)\)` -> `VTData("str\1")`
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *argData = nullptr;
    DTPos *resGrpIdsExp = nullptr;
    DTPos *resReprPosExp = nullptr;

    // Empty input data.
    SECTION("empty argData") {
        argData = DataObjectFactory::create<DTData>(0, false);
        resGrpIdsExp = DataObjectFactory::create<DTPos>(0, false);
        resReprPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    // Non-empty input data, one distinct value.
    SECTION("non-empty argData (one distinct value)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str1.1")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0)});
    }
    // Non-empty input data, unique values.
    // - The input values could be sorted or unsorted.
    SECTION("non-empty argData (unique values, sorted)") {
        argData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str2.2"), VTData("str3.3"), VTData("str4.4"), VTData("str5.5")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values, unsorted)") {
        argData = genGivenVals<DTData>(
            {VTData("str3.3"), VTData("str1.1"), VTData("str2.2"), VTData("str5.5"), VTData("str4.4")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // Non-empty input data, a few distinct values.
    // - The input values could be
    //   - sorted
    //   - unsorted and clustered (all occurrences of each distinct value in a contiguous subsequence)
    //   - unsorted and unclustered (occurrences of a distict value can be separated from each other)
    SECTION("non-empty argData (a few distinct values, sorted)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str3.3"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str4.4")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(1), VTPos(2), VTPos(2), VTPos(2), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(2), VTPos(3), VTPos(6)});
    }
    SECTION("non-empty argData (a few distinct values, unsorted+clustered)") {
        argData = genGivenVals<DTData>({VTData("str2.2"), VTData("str1.1"), VTData("str1.1"), VTData("str4.4"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str3.3")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(1), VTPos(2), VTPos(3), VTPos(3), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (a few distinct values, unsorted+unclustered)") {
        argData = genGivenVals<DTData>({VTData("str2.2"), VTData("str3.3"), VTData("str1.1"), VTData("str4.4"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str1.1")});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(1), VTPos(1), VTPos(2)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3)});
    }

    DTPos *resGrpIdsFnd = nullptr;
    DTPos *resReprPosFnd = nullptr;
    colGroupFirst(resGrpIdsFnd, resReprPosFnd, argData, nullptr);
    CHECK(*resGrpIdsFnd == *resGrpIdsExp);
    CHECK(*resReprPosFnd == *resReprPosExp);

    DataObjectFactory::destroy(argData, resGrpIdsFnd, resReprPosFnd, resGrpIdsExp, resReprPosExp);
}

// There are no invalid input data for the colGroupFirst-kernel, so no tests with invalid data here.