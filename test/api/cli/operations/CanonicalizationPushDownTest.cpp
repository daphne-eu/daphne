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

#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

const std::string dirPath = "test/api/cli/operations/";

#define MAKE_TEST_CASE(name, count)                                                                                    \
    TEST_CASE(name, TAG_OPERATIONS) {                                                                                  \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { compareDaphneToRefSimple(dirPath, name, i); }                \
        }                                                                                                              \
    }

// These tests simply ensure that basic functionality stays
// functional after pushdown optimizations are performed.
// Check the dedicated "pushdown_arithmetics" testcase for more in depth testing
// of the canonicalizations.
MAKE_TEST_CASE("fillEwAdd", 2)
MAKE_TEST_CASE("fillEwSub", 2)
MAKE_TEST_CASE("fillEwMul", 2)
MAKE_TEST_CASE("fillEwDiv", 2)
MAKE_TEST_CASE("fillEwPow", 2)
MAKE_TEST_CASE("fillEwMod", 2)
MAKE_TEST_CASE("fillEwLog", 2)
MAKE_TEST_CASE("fillEwAbs", 2)
MAKE_TEST_CASE("fillEwSign", 2)
MAKE_TEST_CASE("fillEwExp", 2)
MAKE_TEST_CASE("fillEwLn", 1)
MAKE_TEST_CASE("fillEwSqrt", 1)
MAKE_TEST_CASE("randEwAdd", 1)
MAKE_TEST_CASE("randEwSub", 1)
MAKE_TEST_CASE("randEwMul", 1)
MAKE_TEST_CASE("randEwDiv", 1)
MAKE_TEST_CASE("randEwAbs", 2)
MAKE_TEST_CASE("seqEwAdd", 1)
MAKE_TEST_CASE("seqEwSub", 1)
MAKE_TEST_CASE("seqEwMul", 1)
MAKE_TEST_CASE("seqEwDiv", 1)
