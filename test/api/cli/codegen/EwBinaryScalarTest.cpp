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

void test_binary_lowering(const std::string op,
                          const std::string kernel_call,
                          const std::string lowering,
                          const std::string result) {
    std::stringstream out;
    std::stringstream err;

    int status = runDaphne(out, err, "--explain", "llvm", (dirPath + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), Catch::Contains(kernel_call));
    CHECK_THAT(err.str(), !Catch::Contains(lowering));
    CHECK(out.str() == result);

    out.str(std::string());
    err.str(std::string());

    status = runDaphne(out, err, "--explain", "llvm", "--mlir-codegen", (dirPath + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), !Catch::Contains(kernel_call));
    CHECK_THAT(err.str(), Catch::Contains(lowering));
    CHECK(out.str() == result);
}

TEST_CASE("ewBinaryAddScalar", TAG_CODEGEN) {
    test_binary_lowering("add", "llvm.call @_ewAdd__", "llvm.add", "3\n");
}

TEST_CASE("ewBinarySubScalar", TAG_CODEGEN) {
    test_binary_lowering("sub", "llvm.call @_ewSub__", "llvm.sub", "-1\n");
}

TEST_CASE("ewBinaryMulScalar", TAG_CODEGEN) {
    test_binary_lowering("mul", "llvm.call @_ewMul__", "llvm.mul", "2\n");
}

TEST_CASE("ewBinaryDivScalar", TAG_CODEGEN) {
    test_binary_lowering("div", "llvm.call @_ewDiv__", "llvm.fdiv", "1.5\n");
}

TEST_CASE("ewBinaryPowScalar", TAG_CODEGEN) {
    test_binary_lowering("pow", "llvm.call @_ewPow__", "llvm.intr.pow", "9\n");
}

TEST_CASE("ewBinaryAbsScalar", TAG_CODEGEN) {
    test_binary_lowering("abs", "llvm.call @_ewAbs__", "llvm.intr.fabs", "4\n");
}
