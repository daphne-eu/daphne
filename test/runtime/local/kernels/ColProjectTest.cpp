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
#include <runtime/local/kernels/ColProject.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

#define TEST_NAME "ColProject"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t
#define STR_VALUE_TYPES std::string

// TODO make #cols in genGivenVals optional (but still consistent with the other data types, maybe take the #elems as
// #rows, if #rows not specified)

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, numeric data", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTPos *rhsPos = nullptr;
    DTData *resDataExp = nullptr;

    // Empty input positions.
    // - The input data could be empty or non-empty, but the concrete input data doesn't matter.
    SECTION("empty data, empty positions") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resDataExp = DataObjectFactory::create<DTData>(0, false);
    }
    SECTION("non-empty data, empty positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resDataExp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input positions.
    // - The input data must be non-empty, but the concrete input data doesn't matter.
    // - The input positions are characterized by the following properties, which can be freely combined:
    //   - They can contain "all" or just a "subset" of the valid positions.
    //   - They can be "unique" (no repetitions) or "non-unique" (repetitions).
    //   - They can be "sorted" or "unsorted".
    SECTION("non-empty data, non-empty (all, unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 1, 2, 3});
        resDataExp = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
    }
    SECTION("non-empty data, non-empty (all, unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 3, 1, 2});
        resDataExp = genGivenVals<DTData>({VTData(100.0), VTData(103.3), VTData(101.1), VTData(102.2)});
    }
    SECTION("non-empty data, non-empty (all, non-unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 1, 1, 2, 3, 3, 3});
        resDataExp = genGivenVals<DTData>(
            {VTData(100.0), VTData(101.1), VTData(101.1), VTData(102.2), VTData(103.3), VTData(103.3), VTData(103.3)});
    }
    SECTION("non-empty data, non-empty (all, non-unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 3, 3, 1, 2, 1, 3});
        resDataExp = genGivenVals<DTData>(
            {VTData(100.0), VTData(103.3), VTData(103.3), VTData(101.1), VTData(102.2), VTData(101.1), VTData(103.3)});
    }
    SECTION("non-empty data, non-empty (subset, unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({2, 3});
        resDataExp = genGivenVals<DTData>({VTData(102.2), VTData(103.3)});
    }
    SECTION("non-empty data, non-empty (subset, unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({3, 2});
        resDataExp = genGivenVals<DTData>({VTData(103.3), VTData(102.2)});
    }
    SECTION("non-empty data, non-empty (subset, non-unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({2, 3, 3, 3});
        resDataExp = genGivenVals<DTData>({VTData(102.2), VTData(103.3), VTData(103.3), VTData(103.3)});
    }
    SECTION("non-empty data, non-empty (subset, non-unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({3, 3, 2, 3});
        resDataExp = genGivenVals<DTData>({VTData(103.3), VTData(103.3), VTData(102.2), VTData(103.3)});
    }

    DTData *resDataFnd = nullptr;
    colProject(resDataFnd, lhsData, rhsPos, nullptr);
    // TODO check that resDataFnd is not nullptr anymore (maybe do that in all kernel test cases)
    CHECK(*resDataFnd == *resDataExp);

    DataObjectFactory::destroy(lhsData, rhsPos, resDataExp, resDataFnd);
}

// This is the same as "valid args, numeric data", just with string-valued input data.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args, string data", TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DTData = TestType;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTPos *rhsPos = nullptr;
    DTData *resDataExp = nullptr;

    // Empty input positions.
    // - The input data could be empty or non-empty, but the concrete input data doesn't matter.
    SECTION("empty data, empty positions") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resDataExp = DataObjectFactory::create<DTData>(0, false);
    }
    SECTION("non-empty data, empty positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resDataExp = DataObjectFactory::create<DTData>(0, false);
    }
    // Non-empty input positions.
    // - The input data must be non-empty, but the concrete input data doesn't matter.
    // - The input positions are characterized by the following properties, which can be freely combined:
    //   - They can contain "all" or just a "subset" of the valid positions.
    //   - They can be "unique" (no repetitions) or "non-unique" (repetitions).
    //   - They can be "sorted" or "unsorted".
    SECTION("non-empty data, non-empty (all, unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({0, 1, 2, 3});
        resDataExp = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
    }
    SECTION("non-empty data, non-empty (all, unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({0, 3, 1, 2});
        resDataExp = genGivenVals<DTData>({"str100.0", "str103.3", "str101.1", "str102.2"});
    }
    SECTION("non-empty data, non-empty (all, non-unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({0, 1, 1, 2, 3, 3, 3});
        resDataExp =
            genGivenVals<DTData>({"str100.0", "str101.1", "str101.1", "str102.2", "str103.3", "str103.3", "str103.3"});
    }
    SECTION("non-empty data, non-empty (all, non-unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({0, 3, 3, 1, 2, 1, 3});
        resDataExp =
            genGivenVals<DTData>({"str100.0", "str103.3", "str103.3", "str101.1", "str102.2", "str101.1", "str103.3"});
    }
    SECTION("non-empty data, non-empty (subset, unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({2, 3});
        resDataExp = genGivenVals<DTData>({"str102.2", "str103.3"});
    }
    SECTION("non-empty data, non-empty (subset, unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({3, 2});
        resDataExp = genGivenVals<DTData>({"str103.3", "str102.2"});
    }
    SECTION("non-empty data, non-empty (subset, non-unique, sorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({2, 3, 3, 3});
        resDataExp = genGivenVals<DTData>({"str102.2", "str103.3", "str103.3", "str103.3"});
    }
    SECTION("non-empty data, non-empty (subset, non-unique, unsorted) positions") {
        lhsData = genGivenVals<DTData>({"str100.0", "str101.1", "str102.2", "str103.3"});
        rhsPos = genGivenVals<DTPos>({3, 3, 2, 3});
        resDataExp = genGivenVals<DTData>({"str103.3", "str103.3", "str102.2", "str103.3"});
    }

    DTData *resDataFnd = nullptr;
    colProject(resDataFnd, lhsData, rhsPos, nullptr);
    // TODO check that resDataFnd is not nullptr anymore (maybe do that in all kernel test cases)
    CHECK(*resDataFnd == *resDataExp);

    DataObjectFactory::destroy(lhsData, rhsPos, resDataExp, resDataFnd);
}

// We only use numeric value types for the input data here, since this test case is mainly about the input positions and
// the basic functionality for string value types has been tested above.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": invalid args, unsigned positions", TAG_KERNELS, (DATA_TYPES),
                           (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = size_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTPos *rhsPos = nullptr;

    // Empty input data, any non-empty input positions are invalid.
    SECTION("empty data, non-empty positions") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsPos = genGivenVals<DTPos>({0});
    }
    // Non-empty input data, invalid non-empty positions.
    // - The input data are non-empty, but the concrete input data doesn't matter.
    // - The invalid input positions could be characterized by the following properties, which can be freely combined:
    //   - The positions could be "one-too-high" or "far-too-high".
    //   - There could be "1-of-1 invalid" position or "1-of-many" invalid positions (other options are possible, too,
    //     but we don't check them here).
    SECTION("non-empty data, invalid non-empty (one-too-high, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({4});
    }
    SECTION("non-empty data, invalid non-empty (one-too-high, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 4, 2});
    }
    SECTION("non-empty data, invalid non-empty (far-too-high, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({1000000});
    }
    SECTION("non-empty data, invalid non-empty (far-too-high, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 1000000, 2});
    }

    DTData *resDataFnd = nullptr;
    CHECK_THROWS(colProject(resDataFnd, lhsData, rhsPos, nullptr));

    DataObjectFactory::destroy(lhsData, rhsPos);
    if (resDataFnd)
        DataObjectFactory::destroy(resDataFnd);
}

// We only use numeric value types for the input data here, since this test case is mainly about the input positions and
// the basic functionality for string value types has been tested above.
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": invalid args, signed positions", TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DTData = TestType;
    using VTData = typename DTData::VT;

    using VTPos = ssize_t;
    using DTPos = typename TestType::template WithValueType<VTPos>;

    DTData *lhsData = nullptr;
    DTPos *rhsPos = nullptr;

    // Empty input data, any non-empty input positions are invalid.
    SECTION("empty data, non-empty positions") {
        lhsData = DataObjectFactory::create<DTData>(0, false);
        rhsPos = genGivenVals<DTPos>({0});
    }
    // Non-empty input data, invalid non-empty positions.
    // - The input data are non-empty, but the concrete input data doesn't matter.
    // - The invalid input positions could be characterized by the following properties, which can be freely combined:
    //   - The positions could be "one-too-high", "far-too-high" or "one-too-low" (negative, -1), or "far-too-low"
    //   (negative).
    //   - There could be "1-of-1 invalid" position or "1-of-many" invalid positions (other options are possible, too,
    //     but we don't check them here).
    SECTION("non-empty data, invalid non-empty (one-too-high, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({4});
    }
    SECTION("non-empty data, invalid non-empty (one-too-high, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 4, 2});
    }
    SECTION("non-empty data, invalid non-empty (far-too-high, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({1000000});
    }
    SECTION("non-empty data, invalid non-empty (far-too-high, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, 1000000, 2});
    }
    SECTION("non-empty data, invalid non-empty (one-too-low, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({-1});
    }
    SECTION("non-empty data, invalid non-empty (one-too-low, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, -1, 2});
    }
    SECTION("non-empty data, invalid non-empty (far-too-low, 1-of-1) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({-1000000});
    }
    SECTION("non-empty data, invalid non-empty (far-too-low, 1-of-many) positions") {
        lhsData = genGivenVals<DTData>({VTData(100.0), VTData(101.1), VTData(102.2), VTData(103.3)});
        rhsPos = genGivenVals<DTPos>({0, -1000000, 2});
    }

    DTData *resDataFnd = nullptr;
    CHECK_THROWS(colProject(resDataFnd, lhsData, rhsPos, nullptr));

    DataObjectFactory::destroy(lhsData, rhsPos);
    if (resDataFnd)
        DataObjectFactory::destroy(resDataFnd);
}