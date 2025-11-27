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

TEST_CASE("aggDim sum", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x1, double)\n"
                            "6\n"
                            "15\n"
                        "DenseMatrix(2x1, int64_t)\n"
                            "6\n"
                            "15\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "6\n"
                            "15\n"
                        "DenseMatrix(1x3, double)\n"
                            "5 7 9\n"
                        "DenseMatrix(1x3, int64_t)\n"
                            "5 7 9\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "5 7 9\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "aggdim_sum.daphne");
    compareDaphneToStr(result, dirPath + "aggdim_sum.daphne", "--codegen");
}

TEST_CASE("aggDim min", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x1, double)\n"
                            "1\n"
                            "4\n"
                        "DenseMatrix(2x1, int64_t)\n"
                            "1\n"
                            "4\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "1\n"
                            "4\n"
                        "DenseMatrix(1x3, double)\n"
                            "1 2 3\n"
                        "DenseMatrix(1x3, int64_t)\n"
                            "1 2 3\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "1 2 3\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "aggdim_min.daphne");
    compareDaphneToStr(result, dirPath + "aggdim_min.daphne", "--codegen");
}

TEST_CASE("aggDim max", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x1, double)\n"
                            "3\n"
                            "6\n"
                        "DenseMatrix(2x1, int64_t)\n"
                            "3\n"
                            "6\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "3\n"
                            "6\n"
                        "DenseMatrix(1x3, double)\n"
                            "4 5 6\n"
                        "DenseMatrix(1x3, int64_t)\n"
                            "4 5 6\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "4 5 6\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "aggdim_max.daphne");
    compareDaphneToStr(result, dirPath + "aggdim_max.daphne", "--codegen");
}

TEST_CASE("aggDim argMin", TAG_CODEGEN) {
    // clang-format off
    // [1, 2, 1
    //  6, 1, 4]
    std::string result = "DenseMatrix(2x1, uint64_t)\n"
                            "0\n"
                            "1\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "0\n"
                            "1\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "0\n"
                            "1\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "0 1 0\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "0 1 0\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "0 1 0\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "aggdim_argmin.daphne");
    compareDaphneToStr(result, dirPath + "aggdim_argmin.daphne", "--codegen");
}

TEST_CASE("aggDim argMax", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x1, uint64_t)\n"
                            "1\n"
                            "0\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "1\n"
                            "0\n"
                        "DenseMatrix(2x1, uint64_t)\n"
                            "1\n"
                            "0\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "1 0 1\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "1 0 1\n"
                        "DenseMatrix(1x3, uint64_t)\n"
                            "1 0 1\n";
    // clang-format on
    compareDaphneToStr(result, dirPath + "aggdim_argmax.daphne");
    compareDaphneToStr(result, dirPath + "aggdim_argmax.daphne", "--codegen");
}
