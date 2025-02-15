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

#include <api/cli/Utils.h>
#include <parser/metadata/MetaDataParser.h>
#include <tags.h>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/api/cli/io/";
TEST_CASE("evalFrameFromCSVBinOpt", TAG_IO) {
    std::string filename = dirPath + "csv_data1.csv";
    std::filesystem::remove(filename + ".posmap");
    std::filesystem::remove(filename + ".dbdf");
    // build binary file and positional map on first read
    compareDaphneToRef(dirPath + "testReadFrame.txt", dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
    REQUIRE(std::filesystem::exists(filename + ".posmap"));
    REQUIRE(std::filesystem::exists(filename + ".dbdf"));
    std::filesystem::remove(filename + ".posmap");
    compareDaphneToRef(dirPath + "testReadFrame.txt", dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
}

TEST_CASE("evalFrameFromCSVPosMap", TAG_IO) {
    std::string filename = dirPath + "csv_data1.csv";
    std::filesystem::remove(filename + ".posmap");
    std::filesystem::remove(filename + ".dbdf");
    // build binary file and positional map on first read
    compareDaphneToRef(dirPath + "testReadFrame.txt", dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
    REQUIRE(std::filesystem::exists(filename + ".posmap"));
    REQUIRE(std::filesystem::exists(filename + ".dbdf"));
    std::filesystem::remove(filename + ".dbdf");
    compareDaphneToRef(dirPath + "testReadFrame.txt", dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
}