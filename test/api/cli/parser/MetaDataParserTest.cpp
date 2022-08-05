/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <tags.h>

#include <catch.hpp>

#include <parser/metadata/MetaDataParser.h>

#include <iostream>

const std::string dirPath = "test/api/cli/parser/metadataFiles/";

TEST_CASE("Proper meta data file for Matrix", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData1.json";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Proper meta data file for Frame", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData2.json";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file mising \"numRows\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData3.json";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file mising \"numCols\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData4.json";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Matrix meta data file missing \"valueType\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData5.json";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file without \"numNonZeros\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData6.json";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("A non existing meta data file passed to the method", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaMetaData.json";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}