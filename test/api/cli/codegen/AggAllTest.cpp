/*
 * Copyright 2023 The DAPHNE Consortium
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

const std::string dirPath = "test/api/cli/codegen/";

void testAggAllResult(const std::string result, const std::string op) {
    compareDaphneToStr(result, dirPath + "aggall_" + op + ".daphne");
    compareDaphneToStr(result, dirPath + "aggall_" + op + ".daphne", "--mlir-codegen");
}

TEST_CASE("aggAll sum", TAG_CODEGEN) { testAggAllResult("100\n100\n100\n", "sum"); }
TEST_CASE("aggAll min", TAG_CODEGEN) { testAggAllResult("1\n1\n1\n", "min"); }
TEST_CASE("aggAll max", TAG_CODEGEN) { testAggAllResult("6\n6\n6\n", "max"); }
