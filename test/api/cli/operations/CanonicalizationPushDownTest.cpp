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
MAKE_TEST_CASE("pushdownFillEwAdd", 2)
MAKE_TEST_CASE("pushdownFillEwSub", 2)
MAKE_TEST_CASE("pushdownFillEwMul", 2)
MAKE_TEST_CASE("pushdownFillEwDiv", 2)
MAKE_TEST_CASE("pushdownFillEwPow", 2)
MAKE_TEST_CASE("pushdownFillEwMod", 2)
MAKE_TEST_CASE("pushdownFillEwLog", 2)
MAKE_TEST_CASE("pushdownFillEwAbs", 2)
MAKE_TEST_CASE("pushdownFillEwSign", 2)
MAKE_TEST_CASE("pushdownFillEwExp", 2)
MAKE_TEST_CASE("pushdownFillEwLn", 1)
MAKE_TEST_CASE("pushdownFillEwSqrt", 1)
MAKE_TEST_CASE("pushdownRandEwAdd", 1)
MAKE_TEST_CASE("pushdownRandEwSub", 1)
MAKE_TEST_CASE("pushdownRandEwMul", 1)
MAKE_TEST_CASE("pushdownRandEwDiv", 1)
MAKE_TEST_CASE("pushdownRandEwPow", 1)
MAKE_TEST_CASE("pushdownRandEwLog", 1)
MAKE_TEST_CASE("pushdownRandEwAbs", 2)
MAKE_TEST_CASE("pushdownRandEwExp", 2)
MAKE_TEST_CASE("pushdownRandEwLn", 1)
MAKE_TEST_CASE("pushdownRandEwSqrt", 1)
