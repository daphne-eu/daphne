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

#include "run_tests.h"

#include "api/cli/StatusCode.h"
#include "api/cli/Utils.h"

#include <tags.h>

const std::string dirPath = "test/codegen/";

// Place all test files with FileCheck directives in the dirPath.
// LIT will test all *.mlir files in the directory.
TEST_CASE("codegen", TAG_CODEGEN) {
    std::stringstream out;
    std::stringstream err;

    int status = runLIT(out, err, dirPath);

#ifndef NDEBUG
    spdlog::info("runLIT return status: " + std::to_string(status));
    spdlog::info("runLIT out:\n" + out.str());
    spdlog::info("runLIT err:\n" + err.str());
#endif
    CHECK(status == StatusCode::SUCCESS);
}
