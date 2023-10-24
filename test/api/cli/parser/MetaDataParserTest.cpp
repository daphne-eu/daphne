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

#include "run_tests.h"

#include <tags.h>

#include <catch.hpp>

#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/Frame.h>

#include <iostream>

const std::string dirPath = "test/api/cli/parser/metadataFiles/";

TEST_CASE("Proper meta data file for Matrix", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData1";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Proper meta data file for Frame", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData2";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file mising \"numRows\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData3";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file mising \"numCols\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData4";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Matrix meta data file missing \"valueType\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData5";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Meta data file without \"numNonZeros\" key", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData6";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("A non existing meta data file passed to the method", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaMetaData";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Frame meta data file without \"label\" keys", TAG_PARSER)
{
    const std::string metaDataFile = dirPath + "MetaData7";
    REQUIRE_THROWS(MetaDataParser::readMetaData(metaDataFile));
}

TEST_CASE("Frame meta data file with default \"valueType\"", TAG_PARSER)
{
    auto dctx = setupContextAndLogger();
    const std::string metaDataFile = dirPath + "MetaData8";
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFile));
}

TEMPLATE_PRODUCT_TEST_CASE("Write proper meta data file for Matrix", TAG_PARSER,(DenseMatrix, CSRMatrix), (double))
{
    using DT = TestType;
    
    const std::filesystem::path metaDataFile(dirPath + "WriteMatrixMetaData.meta");
    const std::filesystem::path metaDataFileNoSuffix(dirPath + "WriteMatrixMetaData");

    auto m = genGivenVals<DT>(3, {
            0, 0, 1, 0,
            0, 0, 0, 0,
            0, 2, 0, 0,
    });
    
    FileMetaData metaData(m->getNumRows(), m->getNumCols(), true, ValueTypeUtils::codeFor<typename DT::VT>);
    MetaDataParser::writeMetaData(metaDataFileNoSuffix, metaData);
    
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFileNoSuffix));

    // cleanup
    if(std::filesystem::exists(metaDataFile)) {
        std::filesystem::remove(metaDataFile);
    }
}

TEST_CASE("Write proper meta data file for Frame", TAG_PARSER)
{
    const std::filesystem::path metaDataFile(dirPath + "WriteFrameMetaData.meta");
    // the writeMataData method adds the .meta suffix so we need one version here without it
    const std::filesystem::path metaDataFileNoSuffix(dirPath + "WriteFrameMetaData");

    std::vector<ValueTypeCode> schema = {ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels = {"foo", "bar"};
    auto f = DataObjectFactory::create<Frame>(4, 2, schema.data(), labels.data(), false);
    FileMetaData metaData(f->getNumRows(), f->getNumCols(), false, schema, labels, -1);
    
    MetaDataParser::writeMetaData(metaDataFileNoSuffix, metaData);
    
    REQUIRE_NOTHROW(MetaDataParser::readMetaData(metaDataFileNoSuffix));

    // cleanup
    if(std::filesystem::exists(metaDataFile)) {
        std::filesystem::remove(metaDataFile);
    }
}
