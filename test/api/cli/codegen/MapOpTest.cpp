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
#include <sstream>
#include <string>

#include "api/cli/StatusCode.h"

const std::string dirPath = "test/api/cli/codegen/";

TEST_CASE("mapOp", TAG_CODEGEN) {
    std::string result =
        "DenseMatrix(2x2, double)\n"
        "2.1 1\n"
        "6.5 -1.2\n";

    compareDaphneToStr(result, dirPath + "map.daphne");
    compareDaphneToStr(result, dirPath + "map.daphne", "--mlir-codegen", "--no-obj-ref-mgnt");
}

