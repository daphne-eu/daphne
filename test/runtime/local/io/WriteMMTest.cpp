/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/ReadMM.h>
#include <runtime/local/io/WriteMM.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

bool compareContentsFromFile(const std::string &filePath1, const std::string &filePath2) {
    std::ifstream file1(filePath1, std::ios::binary);
    std::ifstream file2(filePath2, std::ios::binary);

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Cannot open one or both files." << std::endl;
        return false;
    }

    std::string line1, line2;
    bool filesAreEqual = true;

    while (std::getline(file1, line1)) {
        if (!std::getline(file2, line2)) {
            filesAreEqual = false;
            break;
        }

        if (line1 != line2) {
            filesAreEqual = false;
            break;
        }
    }

    if (filesAreEqual && std::getline(file2, line2)) {
        if (!line2.empty()) {
            filesAreEqual = false;
        }
    }

    file1.close();
    file2.close();

    return filesAreEqual;
}

TEMPLATE_PRODUCT_TEST_CASE("WriteMM AIG", TAG_IO, (DenseMatrix), (int32_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 4;
    size_t numCols = 3;

    char filename[] = "./test/runtime/local/io/aig.mtx";
    char resultPath[] = "out.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    writeMM(m, resultPath);

    CHECK(compareContentsFromFile(filename, resultPath));
    std::filesystem::remove(resultPath);

    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(1, 0) == 2);
    CHECK(m->get(0, 1) == 5);
    CHECK(m->get(3, 2) == 12);
    CHECK(m->get(2, 1) == 7);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("WriteMM AIK", TAG_IO, (DenseMatrix), (int32_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 4;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/aik.mtx";
    char resultPath[] = "out.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    writeMM(m, resultPath);

    CHECK(compareContentsFromFile(filename, resultPath));
    std::filesystem::remove(resultPath);

    CHECK(m->get(1, 0) == 1);

    for (size_t r = 0; r < numRows; r++) {
        CHECK(m->get(r, r) == 0);
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->get(r, c) == -m->get(c, r));
    }

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("WriteMM AIS", TAG_IO, (DenseMatrix), (int32_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 3;
    size_t numCols = 3;

    char filename[] = "./test/runtime/local/io/ais.mtx";
    char resultPath[] = "out.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    writeMM(m, resultPath);

    CHECK(compareContentsFromFile(filename, resultPath));
    std::filesystem::remove(resultPath);

    CHECK(m->get(1, 1) == 4);

    for (size_t r = 0; r < numRows; r++)
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->get(r, c) == m->get(c, r));

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("WriteMM CIG (CSR)", TAG_IO, (CSRMatrix), (int32_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 9;
    size_t numCols = 9;

    char filename[] = "./test/runtime/local/io/cig.mtx";
    char resultPath[] = "out.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    writeMM(m, resultPath);

    CHECK(compareContentsFromFile(filename, resultPath));
    std::filesystem::remove(resultPath);

    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(2, 0) == 0);
    CHECK(m->get(3, 4) == 9);
    CHECK(m->get(7, 4) == 4);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("WriteMM CRG (CSR)", TAG_IO, (CSRMatrix), (double)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 497;
    size_t numCols = 507;

    char filename[] = "./test/runtime/local/io/crg.mtx";
    char resultPath[] = "out.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    writeMM(m, resultPath);

    CHECK(compareContentsFromFile(filename, resultPath));
    std::filesystem::remove(resultPath);

    CHECK(m->get(5, 0) == 0.25599762);
    CHECK(m->get(6, 0) == 0.13827993);
    CHECK(m->get(200, 4) == 0.20001954);

    DataObjectFactory::destroy(m);
}