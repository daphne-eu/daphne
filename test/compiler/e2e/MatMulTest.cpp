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

const std::string dirPath = "test/compiler/e2e/";

TEST_CASE("matmul", TAG_CODEGEN) {
    std::stringstream out;
    std::stringstream err;

    int status = runDaphne(out, err, (dirPath + "matmul.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    std::string result =
        "DenseMatrix(3x3, double)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";
    CHECK(out.str() == result);

    out.str(std::string());
    err.str(std::string());

    status = runDaphne(out, err, "--codegen", (dirPath + "matmul.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == result);
}


TEST_CASE("matvec", TAG_CODEGEN) {
    std::stringstream out;
    std::stringstream err;

    int status = runDaphne(out, err, (dirPath + "matvec.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    std::string result =
        "DenseMatrix(3x1, double)\n"
        "45\n"
        "45\n"
        "45\n";
    CHECK(out.str() == result);

    out.str(std::string());
    err.str(std::string());

    status = runDaphne(out, err, "--codegen", (dirPath + "matvec.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == result);
}
