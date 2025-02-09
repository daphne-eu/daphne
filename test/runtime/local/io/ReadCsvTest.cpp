/*
 * Copyright 2021 The DAPHNE Consortium
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
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cmath>
#include <cstdint>
#include <limits>

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv", TAG_IO, (DenseMatrix), (double)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == -0.2);
    CHECK(m->get(0, 2) == 0.1);
    CHECK(m->get(0, 3) == 0.2);

    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 5.41);
    CHECK(m->get(1, 2) == 6.22216);
    CHECK(m->get(1, 3) == 5);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, densematrix of doubles using binary optimization", "[TAG_IO][binOpt]") {
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    DenseMatrix<double>* m_new = nullptr;
    DenseMatrix<double>* m = nullptr;

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    std::cout << "First CSV read for DenseMatrix with binary optimization (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, ReadOpts(true, false, true));
    REQUIRE(std::filesystem::exists(binFile));

    // Verify dimensions and cell values.
    REQUIRE(m_new->getNumRows() == numRows);
    REQUIRE(m_new->getNumCols() == numCols);
    CHECK(m_new->get(0,0) == Approx(-0.1));
    CHECK(m_new->get(0,1) == Approx(-0.2));
    CHECK(m_new->get(0,2) == Approx(0.1));
    CHECK(m_new->get(0,3) == Approx(0.2));
    CHECK(m_new->get(1,0) == Approx(3.14));
    CHECK(m_new->get(1,1) == Approx(5.41));
    CHECK(m_new->get(1,2) == Approx(6.22216));
    CHECK(m_new->get(1,3) == Approx(5));

    std::cout << "Second CSV read for DenseMatrix with binary optimization (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, ReadOpts(true, false, true));

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);
    CHECK(m->get(0,0) == Approx(-0.1));
    CHECK(m->get(0,1) == Approx(-0.2));
    CHECK(m->get(0,2) == Approx(0.1));
    CHECK(m->get(0,3) == Approx(0.2));
    CHECK(m->get(1,0) == Approx(3.14));
    CHECK(m->get(1,1) == Approx(5.41));
    CHECK(m->get(1,2) == Approx(6.22216));
    CHECK(m->get(1,3) == Approx(5));

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

TEST_CASE("ReadCsv, densematrix of doubles using positional map", "[TAG_IO][posMap]") {
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    // Remove any pre-existing positional map.
    std::string posMapFile = std::string(filename) + ".posmap";
    if (std::filesystem::exists(posMapFile))
        std::filesystem::remove(posMapFile);

    DenseMatrix<double>* m_new = nullptr;
    DenseMatrix<double>* m = nullptr;

    std::cout << "First CSV read for DenseMatrix with positional map (writing .posmap file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, ReadOpts(true, true, false));
    REQUIRE(std::filesystem::exists(posMapFile));

    std::cout << "Second CSV read for DenseMatrix with positional map (using .posmap file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, ReadOpts(true, true, false));

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);
    CHECK(m->get(0,0) == Approx(-0.1));
    CHECK(m->get(0,1) == Approx(-0.2));
    CHECK(m->get(0,2) == Approx(0.1));
    CHECK(m->get(0,3) == Approx(0.2));
    CHECK(m->get(1,0) == Approx(3.14));
    CHECK(m->get(1,1) == Approx(5.41));
    CHECK(m->get(1,2) == Approx(6.22216));
    CHECK(m->get(1,3) == Approx(5));

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(posMapFile);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv", TAG_IO, (DenseMatrix), (uint8_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(0, 1) == 2);
    CHECK(m->get(0, 2) == 3);
    CHECK(m->get(0, 3) == 4);

    /* File contains negative numbers. Expect cast to positive */
    CHECK(m->get(1, 0) == 255);
    CHECK(m->get(1, 1) == 254);
    CHECK(m->get(1, 2) == 253);
    CHECK(m->get(1, 3) == 252);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv, col + row ignore", TAG_IO, (DenseMatrix), (int8_t)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 1;
    size_t numCols = 2;

    char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(0, 1) == 2);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv, INF and NAN parsing", TAG_IO, (DenseMatrix), (double)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv3.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == -std::numeric_limits<double>::infinity());
    CHECK(m->get(0, 1) == std::numeric_limits<double>::infinity());
    CHECK(m->get(0, 2) == -std::numeric_limits<double>::infinity());
    CHECK(m->get(0, 3) == std::numeric_limits<double>::infinity());

    CHECK(std::isnan(m->get(1, 0)));
    CHECK(std::isnan(m->get(1, 1)));
    CHECK(std::isnan(m->get(1, 2)));
    CHECK(std::isnan(m->get(1, 3)));

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, frame of floats", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m = NULL;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(0)->get(0, 0) == -0.1);
    CHECK(m->getColumn<double>(1)->get(0, 0) == -0.2);
    CHECK(m->getColumn<double>(2)->get(0, 0) == 0.1);
    CHECK(m->getColumn<double>(3)->get(0, 0) == 0.2);

    CHECK(m->getColumn<double>(0)->get(1, 0) == 3.14);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 5.41);
    CHECK(m->getColumn<double>(2)->get(1, 0) == 6.22216);
    CHECK(m->getColumn<double>(3)->get(1, 0) == 5);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, frame of floats using positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m = NULL;
    Frame *m_new = NULL;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    if(std::filesystem::exists(filename+std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
    std::cout << "first csv read" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    std::cout << "first csv read done" << std::endl;
    REQUIRE(std::filesystem::exists(filename+std::string(".posmap")));
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    std::cout << "second csv read done" << std::endl;

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(0)->get(0, 0) == -0.1);
    CHECK(m->getColumn<double>(1)->get(0, 0) == -0.2);
    CHECK(m->getColumn<double>(2)->get(0, 0) == 0.1);
    CHECK(m->getColumn<double>(3)->get(0, 0) == 0.2);

    CHECK(m->getColumn<double>(0)->get(1, 0) == 3.14);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 5.41);
    CHECK(m->getColumn<double>(2)->get(1, 0) == 6.22216);
    CHECK(m->getColumn<double>(3)->get(1, 0) == 5);

    REQUIRE(m_new->getNumRows() == numRows);
        REQUIRE(m_new->getNumCols() == numCols);

    CHECK(m_new->getColumn<double>(0)->get(0, 0) == -0.1);
    CHECK(m_new->getColumn<double>(1)->get(0, 0) == -0.2);
    CHECK(m_new->getColumn<double>(2)->get(0, 0) == 0.1);
    CHECK(m_new->getColumn<double>(3)->get(0, 0) == 0.2);

    CHECK(m_new->getColumn<double>(0)->get(1, 0) == 3.14);
    CHECK(m_new->getColumn<double>(1)->get(1, 0) == 5.41);
    CHECK(m_new->getColumn<double>(2)->get(1, 0) == 6.22216);
    CHECK(m_new->getColumn<double>(3)->get(1, 0) == 5);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);

    if(std::filesystem::exists(filename+std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of uint8s", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8};
    Frame *m = NULL;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);
    CHECK(m->getColumn<uint8_t>(2)->get(0, 0) == 3);
    CHECK(m->getColumn<uint8_t>(3)->get(0, 0) == 4);

    /* File contains negative numbers. Expect cast to positive */
    CHECK(m->getColumn<uint8_t>(0)->get(1, 0) == 255);
    CHECK(m->getColumn<uint8_t>(1)->get(1, 0) == 254);
    CHECK(m->getColumn<uint8_t>(2)->get(1, 0) == 253);
    CHECK(m->getColumn<uint8_t>(3)->get(1, 0) == 252);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, frame of numbers and strings", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::UI64, ValueTypeCode::F64, ValueTypeCode::STR, ValueTypeCode::UI64,
                              ValueTypeCode::F64};
    Frame *m = NULL;

    size_t numRows = 6;
    size_t numCols = 5;

    char filename[] = "./test/runtime/local/io/ReadCsv5.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<uint64_t>(0)->get(0, 0) == 222);
    CHECK(m->getColumn<uint64_t>(0)->get(1, 0) == 444);
    CHECK(m->getColumn<uint64_t>(0)->get(2, 0) == 555);
    CHECK(m->getColumn<uint64_t>(0)->get(3, 0) == 777);
    CHECK(m->getColumn<uint64_t>(0)->get(4, 0) == 111);
    CHECK(m->getColumn<uint64_t>(0)->get(5, 0) == 222);

    CHECK(m->getColumn<double>(1)->get(0, 0) == 11.5);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 19.3);
    CHECK(m->getColumn<double>(1)->get(2, 0) == 29.9);
    CHECK(m->getColumn<double>(1)->get(3, 0) == 15.2);
    CHECK(m->getColumn<double>(1)->get(4, 0) == 31.8);
    CHECK(m->getColumn<double>(1)->get(5, 0) == 13.9);

    CHECK(m->getColumn<std::string>(2)->get(0, 0) == "world");
    CHECK(m->getColumn<std::string>(2)->get(1, 0) == "sample,");
    CHECK(m->getColumn<std::string>(2)->get(2, 0) == "line1\nline2");
    CHECK(m->getColumn<std::string>(2)->get(3, 0) == "");
    CHECK(m->getColumn<std::string>(2)->get(4, 0) == "\"\\n\\\"abc\"def\\\"");
    CHECK(m->getColumn<std::string>(2)->get(5, 0) == "");

    CHECK(m->getColumn<uint64_t>(3)->get(0, 0) == 444);
    CHECK(m->getColumn<uint64_t>(3)->get(1, 0) == 666);
    CHECK(m->getColumn<uint64_t>(3)->get(2, 0) == 777);
    CHECK(m->getColumn<uint64_t>(3)->get(3, 0) == 999);
    CHECK(m->getColumn<uint64_t>(3)->get(4, 0) == 333);
    CHECK(m->getColumn<uint64_t>(3)->get(5, 0) == 444);

    CHECK(m->getColumn<double>(4)->get(0, 0) == 55.6);
    CHECK(m->getColumn<double>(4)->get(1, 0) == 77.8);
    CHECK(m->getColumn<double>(4)->get(2, 0) == 88.9);
    CHECK(m->getColumn<double>(4)->get(3, 0) == 10.1);
    CHECK(m->getColumn<double>(4)->get(4, 0) == 16.9);
    CHECK(m->getColumn<double>(4)->get(5, 0) == 18.2);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, col + row ignore", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::UI8, ValueTypeCode::UI8};
    Frame *m = NULL;

    size_t numRows = 1;
    size_t numCols = 2;

    char filename[] = "./test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, INF and NAN parsing", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m = NULL;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv3.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(0)->get(0, 0) == -std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(1)->get(0, 0) == std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(2)->get(0, 0) == -std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(3)->get(0, 0) == std::numeric_limits<double>::infinity());

    CHECK(std::isnan(m->getColumn<double>(0)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(1)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(2)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(3)->get(1, 0)));

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadCsv, varying columns", TAG_IO) {
    ValueTypeCode schema[] = {ValueTypeCode::SI8, ValueTypeCode::F32};
    Frame *m = NULL;

    size_t numRows = 2;
    size_t numCols = 2;

    char filename[] = "./test/runtime/local/io/ReadCsv4.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim, schema);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<int8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<float>(1)->get(0, 0) == 0.5);

    CHECK(m->getColumn<int8_t>(0)->get(1, 0) == 2);
    CHECK(m->getColumn<float>(1)->get(1, 0) == 1.0);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadCsv", TAG_IO, (DenseMatrix), (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    DT *m = nullptr;

    size_t numRows = 9;
    size_t numCols = 3;

    char filename[] = "./test/runtime/local/io/ReadCsvStr.csv";
    char delim = ',';

    readCsv(m, filename, numRows, numCols, delim);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == "apple, orange");
    CHECK(m->get(1, 0) == "dog, cat");
    CHECK(m->get(2, 0) == "table");
    CHECK(m->get(3, 0) == "\"");
    CHECK(m->get(4, 0) == "abc\"def");
    CHECK(m->get(5, 0) == "red, blue\\n");
    CHECK(m->get(6, 0) == "\\n\\\"abc\"def\\\"");
    CHECK(m->get(7, 0) == "line1\nline2");
    CHECK(m->get(8, 0) == "\\\"red, \\\"\\\"");

    CHECK(m->get(0, 1) == "35");
    CHECK(m->get(1, 1) == "30");
    CHECK(m->get(2, 1) == "27");
    CHECK(m->get(3, 1) == "22");
    CHECK(m->get(4, 1) == "33");
    CHECK(m->get(5, 1) == "50");
    CHECK(m->get(6, 1) == "28");
    CHECK(m->get(7, 1) == "27");
    CHECK(m->get(8, 1) == "41");

    CHECK(m->get(0, 2) == "Fruit Basket");
    CHECK(m->get(1, 2) == "Pets");
    CHECK(m->get(2, 2) == "Furniture Set");
    CHECK(m->get(3, 2) == "Unknown Item");
    CHECK(m->get(4, 2) == "No Category\\\"");
    CHECK(m->get(5, 2) == "");
    CHECK(m->get(6, 2) == "Mixed string");
    CHECK(m->get(7, 2) == "with newline");
    CHECK(m->get(8, 2) == "");

    DataObjectFactory::destroy(m);
}

// New test cases using positional map optimization

TEST_CASE("ReadCsv, frame of uint8s using positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8, ValueTypeCode::UI8};
    Frame *m = NULL;
    Frame *m_new = NULL;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    REQUIRE(std::filesystem::exists(filename + std::string(".posmap")));
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);
    CHECK(m->getColumn<uint8_t>(2)->get(0, 0) == 3);
    CHECK(m->getColumn<uint8_t>(3)->get(0, 0) == 4);
    CHECK(m->getColumn<uint8_t>(0)->get(1, 0) == 255);
    CHECK(m->getColumn<uint8_t>(1)->get(1, 0) == 254);
    CHECK(m->getColumn<uint8_t>(2)->get(1, 0) == 253);
    CHECK(m->getColumn<uint8_t>(3)->get(1, 0) == 252);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of numbers and strings using positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::UI64, ValueTypeCode::F64, ValueTypeCode::STR, ValueTypeCode::UI64, ValueTypeCode::F64};
    Frame *m = NULL;
    Frame *m_new = NULL;
    size_t numRows = 6;
    size_t numCols = 5;
    char filename[] = "test/runtime/local/io/ReadCsv6.csv";
    char delim = ',';

    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    REQUIRE(std::filesystem::exists(filename + std::string(".posmap")));
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    CHECK(m->getColumn<uint64_t>(0)->get(0, 0) == 222);
    CHECK(m->getColumn<uint64_t>(0)->get(1, 0) == 444);
    CHECK(m->getColumn<uint64_t>(0)->get(2, 0) == 555);
    CHECK(m->getColumn<uint64_t>(0)->get(3, 0) == 777);
    CHECK(m->getColumn<uint64_t>(0)->get(4, 0) == 111);
    CHECK(m->getColumn<uint64_t>(0)->get(5, 0) == 222);

    CHECK(m->getColumn<double>(1)->get(0, 0) == 11.5);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 19.3);
    CHECK(m->getColumn<double>(1)->get(2, 0) == 29.9);
    CHECK(m->getColumn<double>(1)->get(3, 0) == 15.2);
    CHECK(m->getColumn<double>(1)->get(4, 0) == 31.8);
    CHECK(m->getColumn<double>(1)->get(5, 0) == 13.9);

    CHECK(m->getColumn<std::string>(2)->get(0, 0) == "world");
    CHECK(m->getColumn<std::string>(2)->get(1, 0) == "sample,");
    CHECK(m->getColumn<std::string>(2)->get(2, 0) == "line1line2");//"\n" not working
    CHECK(m->getColumn<std::string>(2)->get(3, 0) == "");
    CHECK(m->getColumn<std::string>(2)->get(4, 0) == "\"\"\\\"abc\"\"def\\\"");//\n removed
    CHECK(m->getColumn<std::string>(2)->get(5, 0) == "");

    CHECK(m->getColumn<uint64_t>(3)->get(0, 0) == 444);
    CHECK(m->getColumn<uint64_t>(3)->get(1, 0) == 666);
    CHECK(m->getColumn<uint64_t>(3)->get(2, 0) == 777);
    CHECK(m->getColumn<uint64_t>(3)->get(3, 0) == 999);
    CHECK(m->getColumn<uint64_t>(3)->get(4, 0) == 333);
    CHECK(m->getColumn<uint64_t>(3)->get(5, 0) == 444);

    CHECK(m->getColumn<double>(4)->get(0, 0) == 55.6);
    CHECK(m->getColumn<double>(4)->get(1, 0) == 77.8);
    CHECK(m->getColumn<double>(4)->get(2, 0) == 88.9);
    CHECK(m->getColumn<double>(4)->get(3, 0) == 10.1);
    CHECK(m->getColumn<double>(4)->get(4, 0) == 16.9);
    CHECK(m->getColumn<double>(4)->get(5, 0) == 18.2);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of INF and NAN parsing using positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m = NULL;
    Frame *m_new = NULL;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv3.csv";
    char delim = ',';

    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    REQUIRE(std::filesystem::exists(filename + std::string(".posmap")));
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    CHECK(m->getColumn<double>(0)->get(0, 0) == -std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(1)->get(0, 0) == std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(2)->get(0, 0) == -std::numeric_limits<double>::infinity());
    CHECK(m->getColumn<double>(3)->get(0, 0) == std::numeric_limits<double>::infinity());
    CHECK(std::isnan(m->getColumn<double>(0)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(1)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(2)->get(1, 0)));
    CHECK(std::isnan(m->getColumn<double>(3)->get(1, 0)));

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of varying columns using positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::SI8, ValueTypeCode::F32};
    Frame *m = NULL;
    Frame *m_new = NULL;
    size_t numRows = 2;
    size_t numCols = 2;
    char filename[] = "test/runtime/local/io/ReadCsv4.csv";
    char delim = ',';

    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));
    REQUIRE(std::filesystem::exists(filename + std::string(".posmap")));
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    CHECK(m->getColumn<int8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<float>(1)->get(0, 0) == 0.5);
    CHECK(m->getColumn<int8_t>(0)->get(1, 0) == 2);
    CHECK(m->getColumn<float>(1)->get(1, 0) == 1.0);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of floats: normal vs positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m_normal = NULL, *m_opt = NULL;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "./test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    // Normal read
    readCsv(m_normal, filename, numRows, numCols, delim, schema);
    // Remove any stale posmap and perform optimized read
    if(std::filesystem::exists(std::string(filename) + ".posmap")) {
        std::filesystem::remove(std::string(filename) + ".posmap");
    }
    readCsv(m_opt, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    // Compare cell values row-wise
    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<double>(0)->get(r, 0) == m_opt->getColumn<double>(0)->get(r, 0));
        CHECK(m_normal->getColumn<double>(1)->get(r, 0) == m_opt->getColumn<double>(1)->get(r, 0));
        CHECK(m_normal->getColumn<double>(2)->get(r, 0) == m_opt->getColumn<double>(2)->get(r, 0));
        CHECK(m_normal->getColumn<double>(3)->get(r, 0) == m_opt->getColumn<double>(3)->get(r, 0));
    }
    DataObjectFactory::destroy(m_normal);
    DataObjectFactory::destroy(m_opt);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of numbers and strings: normal vs positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::UI64, ValueTypeCode::F64, ValueTypeCode::STR, ValueTypeCode::UI64, ValueTypeCode::F64};
    Frame *m_normal = NULL, *m_opt = NULL;
    size_t numRows = 6;
    size_t numCols = 5;
    char filename[] = "./test/runtime/local/io/ReadCsv5.csv";
    char delim = ',';

    // Normal read
    readCsv(m_normal, filename, numRows, numCols, delim, schema);
    // Remove any stale posmap and perform optimized read
    if(std::filesystem::exists(std::string(filename) + ".posmap")) {
        std::filesystem::remove(std::string(filename) + ".posmap");
    }
    readCsv(m_opt, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    // For each row compare all columns explicitly
    // Column 0: UI64
    CHECK(m_normal->getColumn<uint64_t>(0)->get(0, 0) == m_opt->getColumn<uint64_t>(0)->get(0, 0));
    CHECK(m_normal->getColumn<uint64_t>(0)->get(1, 0) == m_opt->getColumn<uint64_t>(0)->get(1, 0));
    CHECK(m_normal->getColumn<uint64_t>(0)->get(2, 0) == m_opt->getColumn<uint64_t>(0)->get(2, 0));
    CHECK(m_normal->getColumn<uint64_t>(0)->get(3, 0) == m_opt->getColumn<uint64_t>(0)->get(3, 0));
    CHECK(m_normal->getColumn<uint64_t>(0)->get(4, 0) == m_opt->getColumn<uint64_t>(0)->get(4, 0));
    CHECK(m_normal->getColumn<uint64_t>(0)->get(5, 0) == m_opt->getColumn<uint64_t>(0)->get(5, 0));
    // Column 1: F64
    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<double>(1)->get(r, 0) == m_opt->getColumn<double>(1)->get(r, 0));
    }
    // Column 2: STR
    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<std::string>(2)->get(r, 0) == m_opt->getColumn<std::string>(2)->get(r, 0));
    }
    // Column 3: UI64
    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<uint64_t>(3)->get(r, 0) == m_opt->getColumn<uint64_t>(3)->get(r, 0));
    }
    // Column 4: F64
    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<double>(4)->get(r, 0) == m_opt->getColumn<double>(4)->get(r, 0));
    }
    DataObjectFactory::destroy(m_normal);
    DataObjectFactory::destroy(m_opt);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of INF and NAN parsing: normal vs positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m_normal = NULL, *m_opt = NULL;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "./test/runtime/local/io/ReadCsv3.csv";
    char delim = ',';

    // Normal read
    readCsv(m_normal, filename, numRows, numCols, delim, schema);
    if(std::filesystem::exists(std::string(filename) + ".posmap")) {
        std::filesystem::remove(std::string(filename) + ".posmap");
    }
    // Optimized read via positional map
    readCsv(m_opt, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    for(size_t r = 0; r < numRows; r++) {
        for(size_t c = 0; c < numCols; c++) {
            double normalVal = m_normal->getColumn<double>(c)->get(r, 0);
            double optVal = m_opt->getColumn<double>(c)->get(r, 0);
            if(r == 1) {
                // Values in row 1 are expected to be NAN
                CHECK(std::isnan(normalVal));
                CHECK(std::isnan(optVal));
            } else {
                CHECK(normalVal == optVal);
            }
        }
    }
    DataObjectFactory::destroy(m_normal);
    DataObjectFactory::destroy(m_opt);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}

TEST_CASE("ReadCsv, frame of varying columns: normal vs positional map", "[TAG_IO][posMap]") {
    ValueTypeCode schema[] = {ValueTypeCode::SI8, ValueTypeCode::F32};
    Frame *m_normal = NULL, *m_opt = NULL;
    size_t numRows = 2;
    size_t numCols = 2;
    char filename[] = "./test/runtime/local/io/ReadCsv4.csv";
    char delim = ',';

    // Normal read
    readCsv(m_normal, filename, numRows, numCols, delim, schema);
    if(std::filesystem::exists(std::string(filename) + ".posmap")) {
        std::filesystem::remove(std::string(filename) + ".posmap");
    }
    // Optimized read via positional map
    readCsv(m_opt, filename, numRows, numCols, delim, schema, ReadOpts(true,true,false));

    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_normal->getColumn<int8_t>(0)->get(r, 0) == m_opt->getColumn<int8_t>(0)->get(r, 0));
        CHECK(m_normal->getColumn<float>(1)->get(r, 0) == m_opt->getColumn<float>(1)->get(r, 0));
    }
    DataObjectFactory::destroy(m_normal);
    DataObjectFactory::destroy(m_opt);
    if(std::filesystem::exists(filename + std::string(".posmap"))) {
        std::filesystem::remove(filename + std::string(".posmap"));
    }
}


// Test case: binary optimization for frame of floats (.daphne expected)
// The first read writes the .daphne file; the second read uses it.
TEST_CASE("ReadCsv, frame of floats using binary optimization", "[TAG_IO][binOpt]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64,
                              ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m_new = nullptr;
    Frame *m = nullptr;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv1.csv";
    char delim = ',';

    // Remove any existing .daphne file.
    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    std::cout << "First CSV read with binary optimization (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));
    REQUIRE(std::filesystem::exists(binFile));
    
    // Verify basic dimensions and cell values.
    REQUIRE(m_new->getNumRows() == numRows);
    REQUIRE(m_new->getNumCols() == numCols);
    CHECK(m_new->getColumn<double>(0)->get(0, 0) == -0.1);
    CHECK(m_new->getColumn<double>(1)->get(0, 0) == -0.2);
    CHECK(m_new->getColumn<double>(2)->get(0, 0) == 0.1);
    CHECK(m_new->getColumn<double>(3)->get(0, 0) == 0.2);
    CHECK(m_new->getColumn<double>(0)->get(1, 0) == 3.14);
    CHECK(m_new->getColumn<double>(1)->get(1, 0) == 5.41);
    CHECK(m_new->getColumn<double>(2)->get(1, 0) == 6.22216);
    CHECK(m_new->getColumn<double>(3)->get(1, 0) == 5);
    
    
    std::cout << "Second CSV read with binary optimization (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));

    // Verify basic dimensions and cell values.
    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);
    CHECK(m->getColumn<double>(0)->get(0, 0) == -0.1);
    CHECK(m->getColumn<double>(1)->get(0, 0) == -0.2);
    CHECK(m->getColumn<double>(2)->get(0, 0) == 0.1);
    CHECK(m->getColumn<double>(3)->get(0, 0) == 0.2);
    CHECK(m->getColumn<double>(0)->get(1, 0) == 3.14);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 5.41);
    CHECK(m->getColumn<double>(2)->get(1, 0) == 6.22216);
    CHECK(m->getColumn<double>(3)->get(1, 0) == 5);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

// Test case: binary optimization for frame of uint8s (.daphne expected)
TEST_CASE("ReadCsv, frame of uint8s using binary optimization", "[TAG_IO][binOpt]") {
    ValueTypeCode schema[] = {ValueTypeCode::UI8, ValueTypeCode::UI8,
                              ValueTypeCode::UI8, ValueTypeCode::UI8};
    Frame *m_new = nullptr;
    Frame *m = nullptr;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv2.csv";
    char delim = ',';

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);
    if (std::filesystem::exists(filename + std::string(".posmap")))
        std::filesystem::remove(filename + std::string(".posmap"));

    std::cout << "First CSV read with binary optimization for uint8s (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true, true, true));
    REQUIRE(std::filesystem::exists(binFile));
    
    REQUIRE(m_new->getNumRows() == numRows);
    REQUIRE(m_new->getNumCols() == numCols);
    CHECK(m_new->getColumn<uint8_t>(0)->get(0, 0) == 1);
    CHECK(m_new->getColumn<uint8_t>(1)->get(0, 0) == 2);
    CHECK(m_new->getColumn<uint8_t>(2)->get(0, 0) == 3);
    CHECK(m_new->getColumn<uint8_t>(3)->get(0, 0) == 4);
    // Negative numbers wrapped around.
    CHECK(m_new->getColumn<uint8_t>(0)->get(1, 0) == 255);
    CHECK(m_new->getColumn<uint8_t>(1)->get(1, 0) == 254);
    CHECK(m_new->getColumn<uint8_t>(2)->get(1, 0) == 253);
    CHECK(m_new->getColumn<uint8_t>(3)->get(1, 0) == 252);
    
    //check if posmap is also created when .daphne is found
    CHECK(std::filesystem::exists(filename + std::string(".posmap")));
    
    std::cout << "Second CSV read with binary optimization for uint8s (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true, true, true));

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);
    CHECK(m->getColumn<uint8_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<uint8_t>(1)->get(0, 0) == 2);
    CHECK(m->getColumn<uint8_t>(2)->get(0, 0) == 3);
    CHECK(m->getColumn<uint8_t>(3)->get(0, 0) == 4);
    // Negative numbers wrapped around.
    CHECK(m->getColumn<uint8_t>(0)->get(1, 0) == 255);
    CHECK(m->getColumn<uint8_t>(1)->get(1, 0) == 254);
    CHECK(m->getColumn<uint8_t>(2)->get(1, 0) == 253);
    CHECK(m->getColumn<uint8_t>(3)->get(1, 0) == 252);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);
    if (std::filesystem::exists(filename + std::string(".posmap")))
        std::filesystem::remove(filename + std::string(".posmap"));
}

// Test case: binary optimization for frame of numbers and strings (.daphne expected)
TEST_CASE("ReadCsv, frame of numbers and strings using binary optimization", "[TAG_IO][binOpt]") {
    ValueTypeCode schema[] = {ValueTypeCode::UI64, ValueTypeCode::F64,
                              ValueTypeCode::STR,  ValueTypeCode::UI64,
                              ValueTypeCode::F64};
    Frame *m_new = nullptr;
    Frame *m = nullptr;
    size_t numRows = 6;
    size_t numCols = 5;
    char filename[] = "test/runtime/local/io/ReadCsv5.csv";
    char delim = ',';

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    std::cout << "First CSV read with binary optimization for numbers/strings (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));
    //daphne files currently dont support strings
    REQUIRE(!std::filesystem::exists(binFile));

    REQUIRE(m_new->getNumRows() == numRows);
    REQUIRE(m_new->getNumCols() == numCols);
    // Test several cells along different columns.
    CHECK(m_new->getColumn<uint64_t>(0)->get(0, 0) == 222);
    CHECK(m_new->getColumn<uint64_t>(0)->get(1, 0) == 444);
    CHECK(m_new->getColumn<uint64_t>(0)->get(2, 0) == 555);
    CHECK(m_new->getColumn<uint64_t>(0)->get(3, 0) == 777);
    CHECK(m_new->getColumn<uint64_t>(0)->get(4, 0) == 111);
    CHECK(m_new->getColumn<uint64_t>(0)->get(5, 0) == 222);
    CHECK(m_new->getColumn<double>(1)->get(0, 0) == 11.5);
    CHECK(m_new->getColumn<double>(1)->get(1, 0) == 19.3);
    CHECK(m_new->getColumn<std::string>(2)->get(0, 0) == "world");
    CHECK(m_new->getColumn<std::string>(2)->get(1, 0) == "sample,");
    CHECK(m_new->getColumn<uint64_t>(3)->get(0, 0) == 444);
    CHECK(m_new->getColumn<double>(4)->get(0, 0) == 55.6);
    
    std::cout << "Second CSV read with binary optimization for numbers/strings (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);
    // Test several cells along different columns.
    CHECK(m->getColumn<uint64_t>(0)->get(0, 0) == 222);
    CHECK(m->getColumn<uint64_t>(0)->get(1, 0) == 444);
    CHECK(m->getColumn<uint64_t>(0)->get(2, 0) == 555);
    CHECK(m->getColumn<uint64_t>(0)->get(3, 0) == 777);
    CHECK(m->getColumn<uint64_t>(0)->get(4, 0) == 111);
    CHECK(m->getColumn<uint64_t>(0)->get(5, 0) == 222);
    CHECK(m->getColumn<double>(1)->get(0, 0) == 11.5);
    CHECK(m->getColumn<double>(1)->get(1, 0) == 19.3);
    CHECK(m->getColumn<std::string>(2)->get(0, 0) == "world");
    CHECK(m->getColumn<std::string>(2)->get(1, 0) == "sample,");
    CHECK(m->getColumn<uint64_t>(3)->get(0, 0) == 444);
    CHECK(m->getColumn<double>(4)->get(0, 0) == 55.6);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

// Test case: binary optimization for frame handling INF and NAN (.daphne expected)
TEST_CASE("ReadCsv, frame of INF and NAN parsing using binary optimization", "[TAG_IO][binOpt]") {
    ValueTypeCode schema[] = {ValueTypeCode::F64, ValueTypeCode::F64,
                              ValueTypeCode::F64, ValueTypeCode::F64};
    Frame *m_new = nullptr;
    Frame *m = nullptr;
    size_t numRows = 2;
    size_t numCols = 4;
    char filename[] = "test/runtime/local/io/ReadCsv3.csv";
    char delim = ',';

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    std::cout << "First CSV read (INF/NAN) with binary optimization (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));
    REQUIRE(std::filesystem::exists(binFile));
    
    std::cout << "Second CSV read (INF/NAN) with binary optimization (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));

    for (size_t c = 0; c < numCols; ++c) {
        double valNew = m_new->getColumn<double>(c)->get(0, 0);
        double val = m->getColumn<double>(c)->get(0, 0);
        if (c % 2 == 0) { // first row: INF variations
            CHECK(val == valNew);
        } else {
            // second row should contain NaN values.
            CHECK(std::isnan(m_new->getColumn<double>(c)->get(1, 0)));
            CHECK(std::isnan(m->getColumn<double>(c)->get(1, 0)));
        }
    }

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

// Test case: binary optimization with varying columns
TEST_CASE("ReadCsv, frame of varying columns using binary optimization", "[TAG_IO][binOpt]") {
    ValueTypeCode schema[] = {ValueTypeCode::SI8, ValueTypeCode::F32};
    Frame *m_new = nullptr;
    Frame *m = nullptr;
    size_t numRows = 2;
    size_t numCols = 2;
    char filename[] = "test/runtime/local/io/ReadCsv4.csv";
    char delim = ',';

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    std::cout << "First CSV read with binary optimization (varying columns, writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));
    REQUIRE(std::filesystem::exists(binFile));

    std::cout << "Second CSV read with binary optimization (varying columns, reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, schema, ReadOpts(true, false, true));

    for(size_t r = 0; r < numRows; r++) {
        CHECK(m_new->getColumn<int8_t>(0)->get(r, 0) == m->getColumn<int8_t>(0)->get(r, 0));
        CHECK(m_new->getColumn<float>(1)->get(r, 0)  == m->getColumn<float>(1)->get(r, 0));
    }

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

TEST_CASE("ReadCsv, CSRMatrix of doubles using binary optimization", "[TAG_IO][csr][binOpt]") {
    // Assume the CSV file "ReadCsvCSR.csv" contains 3 nonzero entries.
    // For example, the matrix is 2x4 with nonzero pattern:
    // row 0: col 1, col 2; row 1: col 3.
    size_t numRows = 2;
    size_t numCols = 4;
    // The file must specify the number of nonzeros explicitly.
    ssize_t numNonZeros = 3;
    char filename[] = "test/runtime/local/io/ReadCsvCSR.csv";
    char delim = ',';

    std::string binFile = std::string(filename) + ".daphne";
    if (std::filesystem::exists(binFile))
        std::filesystem::remove(binFile);

    CSRMatrix<double>* m_new = nullptr;
    CSRMatrix<double>* m = nullptr;

    std::cout << "First CSV read for CSRMatrix with binary optimization (writing .daphne file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, numNonZeros, true, ReadOpts(true, false, true));
    REQUIRE(std::filesystem::exists(binFile));

    // Check basic dimensions
    CHECK(m_new->getNumRows() == numRows);
    CHECK(m_new->getNumCols() == numCols);
    // Verify the CSR arrays. For instance, if the CSV file results in:
    // rowOffsets: [0,2,3]  and colIdxs: [1,2,3] with all nonzeros having value 1.
    size_t* rowOffsets = m_new->getRowOffsets();
    CHECK(rowOffsets[0] == 0);
    CHECK(rowOffsets[1] == 2);
    CHECK(rowOffsets[2] == 3);
    size_t* colIdxs = m_new->getColIdxs();
    double* values = m_new->getValues();
    for (size_t i = 0; i < static_cast<size_t>(numNonZeros); ++i) {
        // Check that each column index is within bounds and each value equals 1.
        CHECK(colIdxs[i] < numCols);
        CHECK(values[i] == 1);
    }

    std::cout << "Second CSV read for CSRMatrix with binary optimization (reading .daphne file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, numNonZeros, true, ReadOpts(true, false, true));

    CHECK(m->getNumRows() == numRows);
    CHECK(m->getNumCols() == numCols);
    size_t* rowOffsets2 = m->getRowOffsets();
    for(size_t i = 0; i <= numRows; i++) {
        CHECK(rowOffsets2[i] == rowOffsets[i]);
    }

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(binFile);
}

TEST_CASE("ReadCsv, CSRMatrix of doubles using positional map", "[TAG_IO][csr][posMap]") {
    size_t numRows = 2;
    size_t numCols = 4;
    ssize_t numNonZeros = 3;
    char filename[] = "test/runtime/local/io/ReadCsvCSR.csv";
    char delim = ',';

    std::string posMapFile = std::string(filename) + ".posmap";
    if (std::filesystem::exists(posMapFile))
        std::filesystem::remove(posMapFile);

    CSRMatrix<double>* m_new = nullptr;
    CSRMatrix<double>* m = nullptr;

    std::cout << "First CSV read for CSRMatrix with positional map (writing .posmap file)" << std::endl;
    readCsv(m_new, filename, numRows, numCols, delim, numNonZeros, true, ReadOpts(true, true, false));
    REQUIRE(std::filesystem::exists(posMapFile));

    std::cout << "Second CSV read for CSRMatrix with positional map (using .posmap file)" << std::endl;
    readCsv(m, filename, numRows, numCols, delim, numNonZeros, true, ReadOpts(true, true, false));

    CHECK(m->getNumRows() == numRows);
    CHECK(m->getNumCols() == numCols);
    // Compare the row offsets from both reads.
    for (size_t i = 0; i <= numRows; i++) {
        CHECK(m->getRowOffsets()[i] == m_new->getRowOffsets()[i]);
    }

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m_new);
    std::filesystem::remove(posMapFile);
}