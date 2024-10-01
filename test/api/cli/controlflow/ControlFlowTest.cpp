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

const std::string dirPath = "test/api/cli/controlflow/";

#define MAKE_TEST_CASE(name, count)                                                                                    \
    TEST_CASE(name, TAG_CONTROLFLOW) {                                                                                 \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { compareDaphneToRefSimple(dirPath, name, i); }                \
        }                                                                                                              \
    }

#define MAKE_FAILURE_TEST_CASE(name, count)                                                                            \
    TEST_CASE(name ", controlflow failure", TAG_CONTROLFLOW) {                                                         \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_" << i << ".daphne") {                                                              \
                std::stringstream out;                                                                                 \
                std::stringstream err;                                                                                 \
                int status = runDaphne(out, err, (dirPath + name + "_" + std::to_string(i) + ".daphne").c_str());      \
                CHECK(status == StatusCode::EXECUTION_ERROR);                                                          \
                if (i == 1) {                                                                                          \
                    CHECK_THAT(out.str(), Catch::Contains("system stopped: unspecified reason"));                      \
                } else {                                                                                               \
                    CHECK_THAT(out.str(), Catch::Contains("system stopped: CUSTOM ERROR"));                            \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

MAKE_TEST_CASE("if", 8)
MAKE_TEST_CASE("for", 23)
MAKE_TEST_CASE("while", 16)
MAKE_TEST_CASE("nested", 26)

MAKE_FAILURE_TEST_CASE("stop", 2)

TEST_CASE("loop-with-many-iterations", TAG_CONTROLFLOW) {
    std::stringstream exp;
    for (size_t i = 1; i <= 500 * 1000; i++)
        exp << i << std::endl;
    compareDaphneToStr(exp.str(), dirPath + "for_manyiterations_1.daphne");
    compareDaphneToStr(exp.str(), dirPath + "while_manyiterations_1.daphne");
}

TEST_CASE("loop-with-many-iterations_variadic-op", TAG_CONTROLFLOW) {
    std::stringstream exp;
    for (size_t i = 1; i <= 500 * 1000; i++)
        exp << "Frame(1x1, [col_0:int64_t])" << std::endl << i << std::endl;
    compareDaphneToStr(exp.str(), dirPath + "for_manyiterations_2.daphne");
    compareDaphneToStr(exp.str(), dirPath + "while_manyiterations_2.daphne");
}