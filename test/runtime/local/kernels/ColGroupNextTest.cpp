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
#include <runtime/local/kernels/ColGroupNext.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#define TEST_NAME "ColGroupNext"
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
    DTPos *argGrpIds = nullptr;
    DTPos *resGrpIdsExp = nullptr;
    DTPos *resReprPosExp = nullptr;

    // Empty input data.

    SECTION("empty argData, empty argGrpIds") {
        argData = DataObjectFactory::create<DTData>(0, false);
        argGrpIds = DataObjectFactory::create<DTPos>(0, false);
        resGrpIdsExp = DataObjectFactory::create<DTPos>(0, false);
        resReprPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    // Non-empty input data.
    // - argData and argGrpIds must always have the same number of rows.
    // - argGrpIds (the group ids of a previous grouping step on another column) could be
    //   - one distinct value
    //   - unqiue values
    //   - multiple distinct values
    //   for each of those, we test various cases of argData.

    // argGrpIds: one distinct value.
    // - The previous grouping (argGrpIds) does not have an impact on the results, i.e., argData alone determines the
    //   results.
    // - We reuse the test cases of the colGroupFirst-kernel.

    // Non-empty input data, argData (one distinct value), argGrpIds (one distinct value).
    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0)});
    }
    // Non-empty input data, argData (unique values), argGrpIds (one distinct value).
    // - The input values could be sorted or unsorted.
    SECTION("non-empty argData (unique values, sorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(3.3), VTData(4.4), VTData(5.5)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values, unsorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData(3.3), VTData(1.1), VTData(2.2), VTData(5.5), VTData(4.4)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // Non-empty input data, argData (multiple distinct values), argGrpIds (one distinct value).
    // - The input values could be
    //   - sorted
    //   - unsorted and clustered (all occurrences of each distinct value in a contiguous subsequence)
    //   - unsorted and unclustered (occurrences of a distict value can be separated from each other)
    SECTION("non-empty argData (multiple distinct values, sorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>(
            {VTData(1.1), VTData(1.1), VTData(2.2), VTData(3.3), VTData(3.3), VTData(3.3), VTData(4.4)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(1), VTPos(2), VTPos(2), VTPos(2), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(2), VTPos(3), VTPos(6)});
    }
    SECTION(
        "non-empty argData (multiple distinct values, unsorted+clustered), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>(
            {VTData(2.2), VTData(1.1), VTData(1.1), VTData(4.4), VTData(3.3), VTData(3.3), VTData(3.3)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(1), VTPos(2), VTPos(3), VTPos(3), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values, unsorted+unclustered), non-empty argGrpIds (one distinct "
            "value)") {
        argData = genGivenVals<DTData>(
            {VTData(2.2), VTData(3.3), VTData(1.1), VTData(4.4), VTData(3.3), VTData(3.3), VTData(1.1)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(1), VTPos(1), VTPos(2)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3)});
    }

    // argGrpIds: unique values.
    // - The resulting grouping is unique, irrespective of the values in argData.
    // - We test just a few cases here: argData have one distinct value, multiple distinct values, or unique values.

    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1), VTData(1.1), VTData(1.1)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(1.1), VTData(2.2), VTData(2.2)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(3.3), VTData(2.2), VTData(4.4), VTData(5.5)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }

    // argGrpIds: multiple distinct values.
    // - The input grouping is refined depending on the values in argData.
    // - argData could have
    //   - one distinct value (no impact on the results)
    //   - one distinct value per input group (no impact on the results)
    //   - multiple distinct values per input group, not shared across input groups (refinement of the input groups)
    //   - multiple distinct values per input group, shared across input groups (refinement of the input groups)
    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(1.1), VTData(1.1), VTData(1.1), VTData(1.1)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(4)});
    }
    SECTION("non-empty argData (one distinct value per input group), non-empty argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(1.1), VTData(1.1), VTData(3.3), VTData(2.2)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values per input group, not shared across input groups), non-empty "
            "argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(3.3), VTData(2.2), VTData(1.1), VTData(4.4), VTData(3.3)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(0), VTPos(3), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values per input group, shared across input groups), non-empty "
            "argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(1.1), VTData(2.2), VTData(1.1), VTData(2.2), VTData(3.3)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(3)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(0), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(4), VTPos(5)});
    }

    DTPos *resGrpIdsFnd = nullptr;
    DTPos *resReprPosFnd = nullptr;
    colGroupNext(resGrpIdsFnd, resReprPosFnd, argData, argGrpIds, nullptr);
    CHECK(*resGrpIdsFnd == *resGrpIdsExp);
    CHECK(*resReprPosFnd == *resReprPosExp);

    DataObjectFactory::destroy(argData, argGrpIds, resGrpIdsFnd, resReprPosFnd, resGrpIdsExp, resReprPosExp);
}

// This is the same as "valid args, numric data", just with string-valued input data (keep consistent).
// - Used the following regex replace: `VTData\((\d+\.\d+)\)` -> `VTData("str\1")`
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *argData = nullptr;
    DTPos *argGrpIds = nullptr;
    DTPos *resGrpIdsExp = nullptr;
    DTPos *resReprPosExp = nullptr;

    // Empty input data.

    SECTION("empty argData, empty argGrpIds") {
        argData = DataObjectFactory::create<DTData>(0, false);
        argGrpIds = DataObjectFactory::create<DTPos>(0, false);
        resGrpIdsExp = DataObjectFactory::create<DTPos>(0, false);
        resReprPosExp = DataObjectFactory::create<DTPos>(0, false);
    }

    // Non-empty input data.
    // - argData and argGrpIds must always have the same number of rows.
    // - argGrpIds (the group ids of a previous grouping step on another column) could be
    //   - one distinct value
    //   - unqiue values
    //   - multiple distinct values
    //   for each of those, we test various cases of argData.

    // argGrpIds: one distinct value.
    // - The previous grouping (argGrpIds) does not have an impact on the results, i.e., argData alone determines the
    //   results.
    // - We reuse the test cases of the colGroupFirst-kernel.

    // Non-empty input data, argData (one distinct value), argGrpIds (one distinct value).
    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str1.1")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0)});
    }
    // Non-empty input data, argData (unique values), argGrpIds (one distinct value).
    // - The input values could be sorted or unsorted.
    SECTION("non-empty argData (unique values, sorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str2.2"), VTData("str3.3"), VTData("str4.4"), VTData("str5.5")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values, unsorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>(
            {VTData("str3.3"), VTData("str1.1"), VTData("str2.2"), VTData("str5.5"), VTData("str4.4")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    // Non-empty input data, argData (multiple distinct values), argGrpIds (one distinct value).
    // - The input values could be
    //   - sorted
    //   - unsorted and clustered (all occurrences of each distinct value in a contiguous subsequence)
    //   - unsorted and unclustered (occurrences of a distict value can be separated from each other)
    SECTION("non-empty argData (multiple distinct values, sorted), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str3.3"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str4.4")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(1), VTPos(2), VTPos(2), VTPos(2), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(2), VTPos(3), VTPos(6)});
    }
    SECTION(
        "non-empty argData (multiple distinct values, unsorted+clustered), non-empty argGrpIds (one distinct value)") {
        argData = genGivenVals<DTData>({VTData("str2.2"), VTData("str1.1"), VTData("str1.1"), VTData("str4.4"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str3.3")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(1), VTPos(2), VTPos(3), VTPos(3), VTPos(3)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values, unsorted+unclustered), non-empty argGrpIds (one distinct "
            "value)") {
        argData = genGivenVals<DTData>({VTData("str2.2"), VTData("str3.3"), VTData("str1.1"), VTData("str4.4"),
                                        VTData("str3.3"), VTData("str3.3"), VTData("str1.1")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0), VTPos(0)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(1), VTPos(1), VTPos(2)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3)});
    }

    // argGrpIds: unique values.
    // - The resulting grouping is unique, irrespective of the values in argData.
    // - We test just a few cases here: argData have one distinct value, multiple distinct values, or unique values.

    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str1.1"), VTData("str1.1"), VTData("str1.1"), VTData("str1.1")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str2.2"), VTData("str1.1"), VTData("str2.2"), VTData("str2.2")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }
    SECTION("non-empty argData (unique values), non-empty argGrpIds (unique values)") {
        argData = genGivenVals<DTData>(
            {VTData("str1.1"), VTData("str3.3"), VTData("str2.2"), VTData("str4.4"), VTData("str5.5")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(3), VTPos(4)});
    }

    // argGrpIds: multiple distinct values.
    // - The input grouping is refined depending on the values in argData.
    // - argData could have
    //   - one distinct value (no impact on the results)
    //   - one distinct value per input group (no impact on the results)
    //   - multiple distinct values per input group, not shared across input groups (refinement of the input groups)
    //   - multiple distinct values per input group, shared across input groups (refinement of the input groups)
    SECTION("non-empty argData (one distinct value), non-empty argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str1.1"), VTData("str1.1"),
                                        VTData("str1.1"), VTData("str1.1")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(4)});
    }
    SECTION("non-empty argData (one distinct value per input group), non-empty argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str2.2"), VTData("str1.1"), VTData("str1.1"),
                                        VTData("str3.3"), VTData("str2.2")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values per input group, not shared across input groups), non-empty "
            "argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str3.3"), VTData("str2.2"), VTData("str1.1"),
                                        VTData("str4.4"), VTData("str3.3")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(1)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(0), VTPos(3), VTPos(1)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(4)});
    }
    SECTION("non-empty argData (multiple distinct values per input group, shared across input groups), non-empty "
            "argGrpIds (multiple distinct values)") {
        argData = genGivenVals<DTData>({VTData("str1.1"), VTData("str1.1"), VTData("str2.2"), VTData("str1.1"),
                                        VTData("str2.2"), VTData("str3.3")});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(0), VTPos(0), VTPos(2), VTPos(3)});
        resGrpIdsExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(0), VTPos(3), VTPos(4)});
        resReprPosExp = genGivenVals<DTPos>({VTPos(0), VTPos(1), VTPos(2), VTPos(4), VTPos(5)});
    }

    DTPos *resGrpIdsFnd = nullptr;
    DTPos *resReprPosFnd = nullptr;
    colGroupNext(resGrpIdsFnd, resReprPosFnd, argData, argGrpIds, nullptr);
    CHECK(*resGrpIdsFnd == *resGrpIdsExp);
    CHECK(*resReprPosFnd == *resReprPosExp);

    DataObjectFactory::destroy(argData, argGrpIds, resGrpIdsFnd, resReprPosFnd, resGrpIdsExp, resReprPosExp);
}

// We only use numeric value types for the input data here, since this test case is mainly about the input sizes and
// the basic functionality for string value types has been tested above.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": invalid args", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = ssize_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *argData = nullptr;
    DTPos *argGrpIds = nullptr;

    // argData and argGrpIds must have the same number of elements.

    // One input empty, the other input non-empty.
    SECTION("empty argData, non-empty argGrpIds") {
        argData = DataObjectFactory::create<DTData>(0, false);
        argGrpIds = genGivenVals<DTPos>({0});
    }
    SECTION("non-empty argData, empty argGrpIds") {
        argData = genGivenVals<DTData>({VTData(1.1)});
        argGrpIds = DataObjectFactory::create<DTPos>(0, false);
    }
    // Both inputs non-empty, but mismatching sizes.
    SECTION("non-empty argData, non-empty argGrpPos (mismatching sizes)") {
        argData = genGivenVals<DTData>({VTData(1.1), VTData(2.2), VTData(3.3)});
        argGrpIds = genGivenVals<DTPos>({VTPos(0), VTPos(1)});
    }

    DTPos *resGrpIdsFnd = nullptr;
    DTPos *resReprPosFnd = nullptr;
    CHECK_THROWS(colGroupNext(resGrpIdsFnd, resReprPosFnd, argData, argGrpIds, nullptr));

    DataObjectFactory::destroy(argData, argGrpIds);
    if (resGrpIdsFnd)
        DataObjectFactory::destroy(resGrpIdsFnd);
    if (resReprPosFnd)
        DataObjectFactory::destroy(resReprPosFnd);
}