/*
 * Copyright 2024 The DAPHNE Consortium
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

const std::string dirPath = "test/api/cli/extensibility/";

#define MAKE_SUCCESS_TEST_CASE(name, count) \
    TEST_CASE(name ", success", TAG_EXTENSIBILITY) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_success_" << i << ".daphne") { \
                compareDaphneToRefSimple(dirPath, name "_success", i); \
            } \
        } \
    }

#define MAKE_IR_TEST_CASE(idx, kernelName) \
    TEST_CASE("repOp_kernel_success_" #idx ".daphne, hint presence", TAG_EXTENSIBILITY) { \
        std::stringstream out; \
        std::stringstream err; \
        int status = runDaphne(out, err, "--enable_property_insert", (dirPath + "rep_op_hint_kernel_success_" #idx ".daphne").c_str()); \
        CHECK(status == StatusCode::SUCCESS); \
    }

// Check if DAPHNE terminates normally when expected and produces the expected output.
MAKE_SUCCESS_TEST_CASE("rep_op_hint_dense", 1)
MAKE_SUCCESS_TEST_CASE("rep_op_hint_csr", 1)