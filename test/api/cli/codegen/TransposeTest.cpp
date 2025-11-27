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

TEST_CASE("transposeOp", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(3x2, double)\n"
                            "1 -4\n"
                            "-2 5\n"
                            "3 -6\n"
                         "DenseMatrix(3x2, int64_t)\n"
                            "1 -4\n"
                            "-2 5\n"
                            "3 -6\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "transpose.daphne");
    compareDaphneToStr(result, dirPath + "transpose.daphne", "--codegen");
}
