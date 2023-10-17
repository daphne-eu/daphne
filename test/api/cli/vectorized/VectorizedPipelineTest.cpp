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

#include <tags.h>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/api/cli/vectorized/";

// TODO Generalize this and make it a reusable utility.
void compareDaphneToDaphneOtherArgs(const std::string &scriptFilePath) {
    // `daphne $scriptFilePath` (no vec, no repr)
    std::stringstream outNN;
    std::stringstream errNN;
    int statusNN = runDaphne(outNN, errNN, scriptFilePath.c_str());
    
    // `daphne --vec $scriptFilePath` (vec, no repr)
    std::stringstream outVN;
    std::stringstream errVN;
    int statusVN = runDaphne(outVN, errVN, "--vec", scriptFilePath.c_str());
    
    // `daphne --vec --select-matrix-repr $scriptFilePath` (vec, repr)
    std::stringstream outVR;
    std::stringstream errVR;
    int statusVR = runDaphne(outVR, errVR, "--vec", "--select-matrix-repr", scriptFilePath.c_str());
    
    // Check if all runs were successful.
    CHECK(statusNN == StatusCode::SUCCESS);
    CHECK(statusVN == StatusCode::SUCCESS);
    CHECK(statusVR == StatusCode::SUCCESS);
    
    // Check if all runs yielded the same output on stdout.
    CHECK(generalizeDataTypes(outNN.str()) == generalizeDataTypes(outVN.str()));
    CHECK(generalizeDataTypes(outNN.str()) == generalizeDataTypes(outVR.str()));
    
    // Check if all runs yielded the same output on stderr.
    CHECK(errNN.str() == errVN.str());
    CHECK(errNN.str() == errVR.str());
}

#define MAKE_TEST_CASE(name, count) \
    TEST_CASE(name, TAG_VECTORIZED) { \
        for(unsigned i = 1; i <= count; i++) { \
            const std::string scriptFilePath = dirPath + name + "_" + std::to_string(i) + ".daphne"; \
            DYNAMIC_SECTION(scriptFilePath) { \
                compareDaphneToDaphneOtherArgs(scriptFilePath); \
            } \
        } \
    }

MAKE_TEST_CASE("pipeline", 7)