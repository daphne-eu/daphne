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

const std::string dirPath = "test/api/cli/sql/";

#define MAKE_SUCCESS_TEST_CASE(name, count) \
    TEST_CASE(name ", success", TAG_SQL) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_success_" << i << ".daphne") { \
                checkDaphneStatusCodeSimple(StatusCode::SUCCESS, dirPath, name "_success", i); \
            } \
        } \
    }

#define MAKE_PARSER_FAILURE_TEST_CASE(name, count) \
    TEST_CASE(name ", parser failure", TAG_SQL) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_parser_failure_" << i << ".daphne") { \
                checkDaphneStatusCodeSimple(StatusCode::PARSER_ERROR, dirPath, name "_parser_failure", i); \
            } \
        } \
    }

#define MAKE_PASS_FAILURE_TEST_CASE(name, count) \
    TEST_CASE(name ", pass failure", TAG_SQL) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_pass_failure_" << i << ".daphne") { \
                checkDaphneStatusCodeSimple(StatusCode::PASS_ERROR, dirPath, name "_pass_failure", i); \
            } \
        } \
    }

#define MAKE_EXEC_FAILURE_TEST_CASE(name, count) \
    TEST_CASE(name ", execution failure", TAG_SQL) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_execution_failure_" << i << ".daphne") { \
                checkDaphneStatusCodeSimple(StatusCode::EXECUTION_ERROR, dirPath, name "_execution_failure", i); \
            } \
        } \
    }

#define MAKE_TEST_CASE(name, count) \
    TEST_CASE(name, TAG_SQL) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { \
                compareDaphneToRefSimple(dirPath, name, i); \
            } \
        } \
    }


MAKE_SUCCESS_TEST_CASE("basic", 4);
MAKE_PASS_FAILURE_TEST_CASE("basic", 3);
MAKE_EXEC_FAILURE_TEST_CASE("basic", 1);

MAKE_SUCCESS_TEST_CASE("cartesian", 2);

MAKE_SUCCESS_TEST_CASE("where", 4);

MAKE_SUCCESS_TEST_CASE("join", 1);

MAKE_SUCCESS_TEST_CASE("group", 3);
MAKE_PASS_FAILURE_TEST_CASE("group", 1);

MAKE_TEST_CASE("group", 4)


MAKE_TEST_CASE("thetaJoin_equal", 4)
MAKE_TEST_CASE("thetaJoin_greaterThan", 2)
MAKE_TEST_CASE("thetaJoin_greaterEqual", 2)
MAKE_TEST_CASE("thetaJoin_lessThan", 2)
MAKE_TEST_CASE("thetaJoin_lessEqual", 2)
MAKE_TEST_CASE("thetaJoin_notEqual", 2)
MAKE_TEST_CASE("thetaJoin_combinedCompare", 2)

MAKE_TEST_CASE("agg_avg", 1)
MAKE_TEST_CASE("agg_count", 1)
MAKE_TEST_CASE("agg_max", 2)
MAKE_TEST_CASE("agg_min", 2)
MAKE_TEST_CASE("agg_multiple", 2)
MAKE_TEST_CASE("agg_sum", 1)

MAKE_TEST_CASE("reuseString", 2)

MAKE_TEST_CASE("select_asterisk", 6)

MAKE_TEST_CASE("distinct", 4)

// TODO Use the scripts testing failure cases.
