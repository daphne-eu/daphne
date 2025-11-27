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

#include "api/cli/StatusCode.h"

const std::string dirPath = "test/api/cli/codegen/";

void test_unary_lowering(const std::string op, const std::string kernel_call, const std::string result) {
    std::stringstream out;
    std::stringstream err;

    int status = runDaphne(out, err, "--explain", "llvm", (dirPath + "ewunary_" + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), Catch::Contains(kernel_call));
    CHECK(out.str() == result);

    out.str(std::string());
    err.str(std::string());

    status =
        runDaphne(out, err, "--explain", "llvm", "--codegen", (dirPath + "ewunary_" + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), !Catch::Contains(kernel_call));
    CHECK(out.str() == result);
}

TEST_CASE("ewUnaryAbs", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "1 2 4\n"
                        "4\n"
                        "DenseMatrix(1x3, int64_t)\n"
                            "1 2 4\n"
                        "4\n";
    // clang-format on
    test_unary_lowering("abs", "llvm.call @_ewAbs__", result);
}

TEST_CASE("ewUnarySqrt", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "1 1.41421 2\n"
                        "2\n";
    // clang-format on
    test_unary_lowering("sqrt", "llvm.call @_ewSqrt__", result);
}

TEST_CASE("ewUnaryExp", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "2.71828 0.135335 54.5982\n"
                        "54.5982\n";
    // clang-format on
    test_unary_lowering("exp", "llvm.call @_ewExp__", result);
}

// TEST_CASE("ewUnaryLn", TAG_CODEGEN) {
//     // clang-format off
//     std::string result = "DenseMatrix(1x3, double)\n"
//                             "0 0.693147 1.38629\n"
//                         "1.38629\n";
//     // clang-format on
//     test_unary_lowering("ln", "llvm.call @_ewLog__", "llvm.intr.log", result);
// }

TEST_CASE("ewUnarySin", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "0.841471 0.909297 -0.756802\n"
                        "-0.756802\n";
    // clang-format on
    test_unary_lowering("sin", "llvm.call @_ewSin__", result);
}

TEST_CASE("ewUnaryCos", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "0.540302 -0.416147 -0.653644\n"
                        "-0.653644\n";
    // clang-format on
    test_unary_lowering("cos", "llvm.call @_ewCos__", result);
}

TEST_CASE("ewUnaryFloor", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "1 -3 4\n"
                        "4\n";
    // clang-format on
    test_unary_lowering("floor", "llvm.call @_ewFloor__", result);
}

TEST_CASE("ewUnaryCeil", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "2 -2 5\n"
                        "5\n";
    // clang-format on
    test_unary_lowering("ceil", "llvm.call @_ewCeil__", result);
}

TEST_CASE("ewUnaryRound", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(1x3, double)\n"
                            "2 -2 5\n"
                        "5\n";
    // clang-format on
    test_unary_lowering("round", "llvm.call @_ewRound__", result);
}
