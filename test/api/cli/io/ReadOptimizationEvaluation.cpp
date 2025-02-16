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

template <typename... Args>
void runDaphneEval( const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());
    
    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(status == StatusCode::SUCCESS);
    std::cout << out.str() << std::endl;
    CHECK(err.str() == "");
}
const std::string dirPath = "test/api/cli/io/";
TEST_CASE("evalFrameFromCSVBinOpt", TAG_IO) {
    std::string filename = dirPath + "csv_data1.csv";
    std::filesystem::remove(filename + ".dbdf");
    // normal read for comparison
    runDaphneEval(dirPath + "evalReadFrame.daphne", "--timing");
    
    // build binary file and positional map on first read
    runDaphneEval(dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
    REQUIRE(std::filesystem::exists(filename + ".dbdf"));
    runDaphneEval(dirPath + "evalReadFrame.daphne", "--timing", "--second-read-opt");
}