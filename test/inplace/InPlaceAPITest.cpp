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
#include <sstream>
#include <regex>

const std::string dirPath = "test/inplace/ref/";

template<typename... Args>
void compareDaphneToRegexErrRef(const std::string &refFilePath, const std::string & scriptFilePath, const std::string & regex, Args ... args) {

    std::string ref = readTextFile(refFilePath);
    
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());

    std::regex pattern(regex);
    
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == ref);
    CHECK(std::regex_search(err.str(), pattern));
}

#define MAKE_TEST_CASE_REGEX(name, suffix, regex, param1, param2) \
    TEST_CASE(std::string(name)+std::string(suffix), TAG_INPLACE) { \
        std::string prefix(dirPath);\
        prefix += (name);\
        compareDaphneToRegexErrRef(prefix + ".txt", prefix + ".daphne", regex, (param1), (param2)); \
    }


/*
* Test cases for the update-in-place feature.
* Checks the output of the compiler to see if the update-in-place flag is set correctly.
*/

MAKE_TEST_CASE_REGEX("bfu-matrix-test", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("lfu-matrix-test", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("rfu-matrix-test", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[false, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("nfu-matrix-test", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[false, false\\]", "--update-in-place", "--explain=update_in_place")

MAKE_TEST_CASE_REGEX("elseif_1", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("elseif_2", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("elseif_3", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[false, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("for_1", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("for_2", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("for_if", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("if_1", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[false, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("if_2", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[false, true\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("if_3", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")
MAKE_TEST_CASE_REGEX("if_4", "", "daphne\\.ewAdd.*inPlaceFutureUse\\s*=\\s*\\[true, false\\]", "--update-in-place", "--explain=update_in_place")