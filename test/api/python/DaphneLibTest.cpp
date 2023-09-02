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

const std::string dirPath = "test/api/python/";

#define MAKE_TEST_CASE(name) \
    TEST_CASE(name, TAG_DAPHNELIB) { \
        DYNAMIC_SECTION(name << ".py") { \
            const std::string prefix = dirPath+name; \
            compareDaphneToDaphneLib(prefix+".py", prefix+".daphne"); \
        } \
    }
#define MAKE_TEST_CASE_SCALAR(name) \
    TEST_CASE(name, TAG_DAPHNELIB) { \
        DYNAMIC_SECTION(name << ".py") { \
            const std::string prefix = dirPath+name; \
            compareDaphneToDaphneLibScalar(prefix+".py", prefix+".daphne"); \
        } \
    }
#define MAKE_TEST_CASE_PARAMETRIZED(name, argument) \
    TEST_CASE((std::string(name)+"_"+std::string(argument)).c_str(), TAG_DAPHNELIB) { \
        DYNAMIC_SECTION(name << ".py") { \
            const std::string prefix = dirPath+name; \
            compareDaphneToDaphneLib(prefix+".py", prefix+".daphne", argument); \
        } \
    }

MAKE_TEST_CASE("data_transfer_numpy_1")
MAKE_TEST_CASE("data_transfer_numpy_2")
MAKE_TEST_CASE("data_transfer_pandas_1")
MAKE_TEST_CASE("random_matrix_generation")
MAKE_TEST_CASE("random_matrix_sum")
MAKE_TEST_CASE("random_matrix_addition")
MAKE_TEST_CASE("random_matrix_subtraction")
MAKE_TEST_CASE("random_matrix_mult")
MAKE_TEST_CASE("random_matrix_div")
MAKE_TEST_CASE("random_matrix_functions")
MAKE_TEST_CASE("scalar_ops")
MAKE_TEST_CASE("frame_cartesian")
MAKE_TEST_CASE_SCALAR("numpy_matrix_ops")
MAKE_TEST_CASE_SCALAR("numpy_matrix_ops_extended")
// Test for DaphneLib complex control flow and lazy evaluated functions
MAKE_TEST_CASE_PARAMETRIZED("if_else_simple", "number=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_simple", "number=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_only_simple", "number=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_only_simple", "number=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_else_2_outputs", "number=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_2_outputs", "number=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_else_complex", "number=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_complex", "number=10.0")
