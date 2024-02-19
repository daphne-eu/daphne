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

TEST_CASE("matmul", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, double)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul.daphne");
    compareDaphneToStr(result, dirPath + "matmul.daphne", "--mlir-codegen");
}
TEST_CASE("matmul vectorized", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, double)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul.daphne", "--mlir-codegen", "--matmul-vec-size-bits=64");
    compareDaphneToStr(result, dirPath + "matmul.daphne", "--mlir-codegen", "--no-obj-ref-mgnt", 
    "--matmul-vec-size-bits=128");
}
TEST_CASE("matmul tiled", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, double)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul.daphne", "--mlir-codegen", "--matmul-fixed-tile-sizes=2,2,2");
}
TEST_CASE("matmul tiled and vectorized", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, double)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul.daphne", "--mlir-codegen", "--matmul-vec-size-bits=64", 
    "--matmul-fixed-tile-sizes=2,2,2");
}
TEST_CASE("matmul single", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, float)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul_single.daphne");
    compareDaphneToStr(result, dirPath + "matmul_single.daphne", "--mlir-codegen");
}
TEST_CASE("matmul si64", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, int64_t)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul_si64.daphne");
    compareDaphneToStr(result, dirPath + "matmul_si64.daphne", "--mlir-codegen");
}
TEST_CASE("matmul ui64", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, uint64_t)\n"
        "45 45 45\n"
        "45 45 45\n"
        "45 45 45\n";

    compareDaphneToStr(result, dirPath + "matmul_ui64.daphne", "--mlir-codegen");
}
TEST_CASE("matmul non square", TAG_CODEGEN TAG_MATMUL) {
    std::string result =
        "DenseMatrix(3x3, double)\n"
        "60 60 60\n"
        "60 60 60\n"
        "60 60 60\n";
    
    compareDaphneToStr(result, dirPath + "matmul_non_square.daphne");
    compareDaphneToStr(result, dirPath + "matmul_non_square.daphne", "--mlir-codegen");
}


TEST_CASE("matvec", TAG_CODEGEN) {
    std::string result =
        "DenseMatrix(3x1, double)\n"
        "45\n"
        "45\n"
        "45\n";

    compareDaphneToStr(result, dirPath + "matvec.daphne");
    compareDaphneToStr(result, dirPath + "matvec.daphne", "--mlir-codegen");
}
