/*
 * Copyright 2025 The DAPHNE Consortium
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

TEST_CASE("ewUnary_abs, sparse", TAG_CODEGEN) {
    std::string result = "CSRMatrix(5x5, double)\n"
                         "0 0 0 0 1\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 1 1 0 0\n"
                         "0 0 0 0 0\n";

    compareDaphneToStr(result, dirPath + "ewunary_abs_sparse.daphne", "--mlir-codegen", "--no-obj-ref-mgnt", "--select-matrix-repr");
}

TEST_CASE("ewBinary_add, sparse", TAG_CODEGEN) {
    std::string result = "DenseMatrix(8x8, double)\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "1 1 1 2 1 1 1 1\n"
                         "1 1 1 1 1 1 1 1\n"
                         "CSRMatrix(8x8, double)\n"
                         "0 0 0 0 0 0 0 0\n"
                         "0 0 0 0 0 0 0 0\n"
                         "0 0 0 0 0 0 0 0\n"
                         "0 0 0 0 0 0 0 0\n"
                         "0 0 0 0 0 0 0 1\n"
                         "0 0 0 0 0 0 0 0\n"
                         "0 0 0 1 0 0 0 0\n"
                         "0 0 0 0 0 0 0 0\n";

    compareDaphneToStr(result, dirPath + "ewbinary_add_sparse.daphne", "--mlir-codegen", "--no-obj-ref-mgnt", "--select-matrix-repr");
}

TEST_CASE("ewBinary_mul, sparse", TAG_CODEGEN) {
    std::string result = "CSRMatrix(5x5, double)\n"
                         "0 0 0 0 2\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 2 2 0 0\n"
                         "0 0 0 0 0\n"
                         "CSRMatrix(5x5, double)\n"
                         "0 0 0 0 1\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 1 1 0 0\n"
                         "0 0 0 0 0\n"
                         "CSRMatrix(5x5, double)\n"
                         "0 0 0 0 1\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n";

    compareDaphneToStr(result, dirPath + "ewbinary_mul_sparse.daphne", "--mlir-codegen", "--no-obj-ref-mgnt", "--select-matrix-repr");
}

TEST_CASE("matmul, sparse-dense", TAG_CODEGEN) {
    std::string result = "DenseMatrix(5x5, double)\n"
                         "1 1 1 1 1\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "2 2 2 2 2\n"
                         "0 0 0 0 0\n"
                         "CSRMatrix(5x5, double)\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "0 0 0 0 0\n"
                         "1 0 0 1 0\n"
                         "0 0 0 0 0\n";

    compareDaphneToStr(result, dirPath + "matmul_sparse.daphne", "--mlir-codegen", "--no-obj-ref-mgnt", "--select-matrix-repr");
}
