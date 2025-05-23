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

#include <string>

const std::string dirPath = "test/api/cli/indexing/";

#define MAKE_SUCCESS_TEST_CASE(name, count)                                                                            \
    TEST_CASE(name ", success", TAG_INDEXING) {                                                                        \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_success_" << i << ".daphne") {                                                      \
                compareDaphneToRefSimple(dirPath, name "_success", i);                                                 \
            }                                                                                                          \
        }                                                                                                              \
    }

#define MAKE_FAILURE_TEST_CASE(name, count)                                                                            \
    TEST_CASE(name ", failure", TAG_INDEXING) {                                                                        \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_failure_" << i << ".daphne") {                                                      \
                checkDaphneFailsSimple(dirPath, name "_failure", i);                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }

MAKE_SUCCESS_TEST_CASE("right_indexing", 6)
MAKE_FAILURE_TEST_CASE("right_indexing", 10)

// TODO Add a test case for multi-assignments (`X[...], Y[...] = ...`).
MAKE_SUCCESS_TEST_CASE("left_indexing", 3)
MAKE_SUCCESS_TEST_CASE("left_indexing_cow", 8)
MAKE_FAILURE_TEST_CASE("left_indexing", 7)