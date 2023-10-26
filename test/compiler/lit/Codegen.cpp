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

const std::string dirPath = "test/compiler/lit/";

// Place all test files with FileCheck directives in the dirPath.
// LIT will test all *.mlir files in the directory.
TEST_CASE("codegen", TAG_CODEGEN) {
    std::stringstream out;
    std::stringstream err;

    int status = 0; // runLIT(out, err, dirPath); TODO: needs docker container update to run during testing

    CHECK(status == StatusCode::SUCCESS);
}
