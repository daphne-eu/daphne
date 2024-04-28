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

#include <filesystem>
#include <string>

const std::string dirPath = "test/api/cli/io/";

TEST_CASE("writeMatrixCSV_Full", TAG_IO) {
    std::string csvPath = dirPath + "matrix_full.csv";
    std::filesystem::remove(csvPath); // remove old file if it still exists
    checkDaphneStatusCode(StatusCode::SUCCESS, dirPath + "writeMatrix_full.daphne", "--args", std::string("outPath=\"" + csvPath + "\"").c_str());
    compareDaphneToRef(dirPath + "matrix_full_ref.csv", dirPath + "readMatrix.daphne", "--args", std::string("inPath=\"" + csvPath + "\"").c_str());
}

TEST_CASE("writeMatrixCSV_View", TAG_IO) {
    std::string csvPath = dirPath + "matrix_view.csv";
    std::filesystem::remove(csvPath); // remove old file if it still exists
    checkDaphneStatusCode(StatusCode::SUCCESS, dirPath + "writeMatrix_view.daphne", "--args", std::string("outPath=\"" + csvPath + "\"").c_str());
    compareDaphneToRef(dirPath + "matrix_view_ref.csv", dirPath + "readMatrix.daphne", "--args", std::string("inPath=\"" + csvPath + "\"").c_str());
}