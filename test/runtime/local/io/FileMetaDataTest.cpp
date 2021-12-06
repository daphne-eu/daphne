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

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/io/FileMetaData.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

TEST_CASE("FileMetaData::ofFile (individual column types, labels given)", TAG_IO) {
    FileMetaData fmd = FileMetaData::ofFile("./test/runtime/local/io/SomeFile0.csv");
    
    CHECK(fmd.numRows == 5);
    REQUIRE(fmd.numCols == 3);
    CHECK_FALSE(fmd.isSingleValueType);
    REQUIRE(fmd.schema.size() == fmd.numCols);
    CHECK(fmd.schema[0] == ValueTypeCode::SI64);
    CHECK(fmd.schema[1] == ValueTypeCode::F64);
    CHECK(fmd.schema[2] == ValueTypeCode::UI32);
    REQUIRE(fmd.labels.size() == fmd.numCols);
    CHECK(fmd.labels[0] == "a");
    CHECK(fmd.labels[1] == "bc");
    CHECK(fmd.labels[2] == "def");
}

TEST_CASE("FileMetaData::ofFile (single value type, labels given)", TAG_IO) {
    FileMetaData fmd = FileMetaData::ofFile("./test/runtime/local/io/SomeFile1.csv");
    
    CHECK(fmd.numRows == 5);
    REQUIRE(fmd.numCols == 3);
    CHECK(fmd.isSingleValueType);
    REQUIRE(fmd.schema.size() == 1);
    CHECK(fmd.schema[0] == ValueTypeCode::F32);
    REQUIRE(fmd.labels.size() == fmd.numCols);
    CHECK(fmd.labels[0] == "a");
    CHECK(fmd.labels[1] == "bc");
    CHECK(fmd.labels[2] == "def");
}

TEST_CASE("FileMetaData::ofFile (individual column types, labels not given)", TAG_IO) {
    FileMetaData fmd = FileMetaData::ofFile("./test/runtime/local/io/SomeFile2.csv");
    
    CHECK(fmd.numRows == 5);
    REQUIRE(fmd.numCols == 3);
    CHECK_FALSE(fmd.isSingleValueType);
    REQUIRE(fmd.schema.size() == fmd.numCols);
    CHECK(fmd.schema[0] == ValueTypeCode::SI64);
    CHECK(fmd.schema[1] == ValueTypeCode::F64);
    CHECK(fmd.schema[2] == ValueTypeCode::UI32);
    REQUIRE(fmd.labels.empty());
}

TEST_CASE("FileMetaData::ofFile (single value type, labels not given)", TAG_IO) {
    FileMetaData fmd = FileMetaData::ofFile("./test/runtime/local/io/SomeFile3.csv");
    
    CHECK(fmd.numRows == 5);
    REQUIRE(fmd.numCols == 3);
    CHECK(fmd.isSingleValueType);
    REQUIRE(fmd.schema.size() == 1);
    CHECK(fmd.schema[0] == ValueTypeCode::F32);
    REQUIRE(fmd.labels.empty());
}

TEST_CASE("FileMetaData::ofFile (single value type, labels given, number of non zeros given)", TAG_IO) {
    FileMetaData fmd = FileMetaData::ofFile("./test/runtime/local/io/SomeFile4.csv");

    CHECK(fmd.numRows == 10);
    REQUIRE(fmd.numCols == 10);
    CHECK(fmd.isSingleValueType);
    REQUIRE(fmd.schema.size() == 1);
    CHECK(fmd.schema[0] == ValueTypeCode::F64);
    REQUIRE(fmd.labels.empty());
    CHECK(fmd.numNonZeros == 2);
}
