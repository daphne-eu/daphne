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

#define MAKE_TEST_CASE(name, count) \
    TEST_CASE(name, TAG_OPERATIONS) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { \
                compareDaphneToRefSimple(dirPath, name, i); \
            } \
        } \
    }

MAKE_TEST_CASE("aggMax", 1)
MAKE_TEST_CASE("aggMin", 1)
MAKE_TEST_CASE("bin", 2)
MAKE_TEST_CASE("cbind", 1)
MAKE_TEST_CASE("createFrame", 1)
MAKE_TEST_CASE("ctable", 1)
MAKE_TEST_CASE("gemv", 1)
MAKE_TEST_CASE("idxMax", 1)
MAKE_TEST_CASE("idxMin", 1)
MAKE_TEST_CASE("mean", 1)
MAKE_TEST_CASE("operator_at", 2)
MAKE_TEST_CASE("operator_minus", 1)
MAKE_TEST_CASE("operator_plus", 2)
MAKE_TEST_CASE("operator_slash", 1)
MAKE_TEST_CASE("operator_times", 1)
MAKE_TEST_CASE("rbind", 1)
MAKE_TEST_CASE("recode", 3)
MAKE_TEST_CASE("replace", 1)
MAKE_TEST_CASE("seq", 1)
MAKE_TEST_CASE("solve", 1)
MAKE_TEST_CASE("sqrt", 1)
MAKE_TEST_CASE("sum", 1)
MAKE_TEST_CASE("syrk", 1)