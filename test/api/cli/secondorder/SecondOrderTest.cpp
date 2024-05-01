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

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/api/cli/secondorder/";

#define MAKE_TEST_CASE(name, count) \
    TEST_CASE(name, TAG_SECONDORDER) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { \
                compareDaphneToRefSimple(dirPath, name, i); \
            } \
        } \
    }

#define MAKE_FAILURE_TEST_CASE(name, count) \
    TEST_CASE(name ", failure", TAG_SECONDORDER) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_failure_" << i << ".daphne") { \
                checkDaphneFailsSimple(dirPath, name "_failure", i); \
            } \
        } \
    }

MAKE_TEST_CASE("map", 3)
MAKE_FAILURE_TEST_CASE("map", 5)