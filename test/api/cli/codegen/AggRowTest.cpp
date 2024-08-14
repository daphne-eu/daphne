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

#include <string>

const std::string dirPath = "test/api/cli/codegen/";

TEST_CASE("aggRow", TAG_CODEGEN) {
    std::string result = 
        "DenseMatrix(3x1, double)\n"
            "10\n"
            "26\n"
            "42\n"
        "DenseMatrix(3x1, float)\n"
            "10\n"
            "26\n"
            "42\n"
        "DenseMatrix(3x1, int64_t)\n"
            "10\n"
            "26\n"
            "42\n";

    compareDaphneToStr(result, dirPath + "sum_aggrow.daphne");
    compareDaphneToStr(result, dirPath + "sum_aggrow.daphne", "--mlir-codegen");
}