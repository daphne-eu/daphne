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
#include <runtime/local/kernels/ColMerge.h>

#include <tags.h>

#include <catch.hpp>

#define TEST_NAME "ColMerge"
#define DATA_TYPES Column
#define VALUE_TYPES int64_t, uint32_t, int8_t, size_t

// The left-hand-side (lhs) and right-hand-side (rhs) input positions for the colMerge-kernel must both be sorted
// and unqiue. For performance reasons, the kernel does not check if this requirement is fulfilled. Thus, we do not test
// if unsorted or non-unqiue inputs are detected. Other than unsorted or non-unique inputs, there are no invalid inputs.
// Thus, we don't test any invalid inputs here.

// The colMerge-kernel is meant to work on positions (not on data). Thus, we only test with integral value types.

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME ": valid args", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DTPos = TestType;

    DTPos *lhsPos = nullptr;
    DTPos *rhsPos = nullptr;
    DTPos *resPosExp = nullptr;

    // At least one empty input.
    SECTION("empty lhsPos, empty rhsPos") {
        lhsPos = DataObjectFactory::create<DTPos>(0, false);
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resPosExp = DataObjectFactory::create<DTPos>(0, false);
    }
    SECTION("empty lhsPos, non-empty rhsPos") {
        lhsPos = DataObjectFactory::create<DTPos>(0, false);
        rhsPos = genGivenVals<DTPos>({0, 1, 3});
        resPosExp = genGivenVals<DTPos>({0, 1, 3});
    }
    SECTION("non-empty lhsPos, empty rhsPos") {
        lhsPos = genGivenVals<DTPos>({0, 1, 3});
        rhsPos = DataObjectFactory::create<DTPos>(0, false);
        resPosExp = genGivenVals<DTPos>({0, 1, 3});
    }
    // Two non-empty inputs, hence non-empty result.
    // - The lhs and rhs positions could be the "same".
    // - The lhs positions could be a subset of the rhs positions ("lhs-in-rhs") or vice versa ("rhs-in-lhs").
    // - All lhs positions could be before all rhs positions ("lhs-before-rhs") or vice versa ("rhs-before-rhs").
    // - None of the inputs could a subset of the other one but they could still be "overlapping".
    SECTION("non-empty lhsPos, non-empty rhsPos (same)") {
        lhsPos = genGivenVals<DTPos>({0, 1, 3});
        rhsPos = genGivenVals<DTPos>({0, 1, 3});
        resPosExp = genGivenVals<DTPos>({0, 1, 3});
    }
    SECTION("non-empty lhsPos, non-empty rhsPos (lhs-in-rhs)") {
        lhsPos = genGivenVals<DTPos>({0, 3});
        rhsPos = genGivenVals<DTPos>({0, 1, 3});
        resPosExp = genGivenVals<DTPos>({0, 1, 3});
    }
    SECTION("non-empty lhsPos, non-empty rhsPos (rhs-in-lhs)") {
        lhsPos = genGivenVals<DTPos>({0, 1, 3});
        rhsPos = genGivenVals<DTPos>({0, 3});
        resPosExp = genGivenVals<DTPos>({0, 1, 3});
    }
    SECTION("non-empty lhsPos, non-empty rhsPos (lhs-before-rhs)") {
        lhsPos = genGivenVals<DTPos>({0, 1, 3});
        rhsPos = genGivenVals<DTPos>({7, 8, 10, 12});
        resPosExp = genGivenVals<DTPos>({0, 1, 3, 7, 8, 10, 12});
    }
    SECTION("non-empty lhsPos, non-empty rhsPos (rhs-before-lhs)") {
        lhsPos = genGivenVals<DTPos>({7, 8, 10, 12});
        rhsPos = genGivenVals<DTPos>({0, 1, 3});
        resPosExp = genGivenVals<DTPos>({0, 1, 3, 7, 8, 10, 12});
    }
    SECTION("non-empty lhsPos, non-empty rhsPos (overlapping)") {
        lhsPos = genGivenVals<DTPos>({0, 3, 8, 10});
        rhsPos = genGivenVals<DTPos>({1, 7, 12});
        resPosExp = genGivenVals<DTPos>({0, 1, 3, 7, 8, 10, 12});
    }

    DTPos *resPosFnd = nullptr;
    colMerge(resPosFnd, lhsPos, rhsPos, nullptr);
    CHECK(*resPosFnd == *resPosExp);

    DataObjectFactory::destroy(lhsPos, rhsPos, resPosExp, resPosFnd);
}