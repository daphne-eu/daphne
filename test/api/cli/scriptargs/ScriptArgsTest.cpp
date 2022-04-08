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

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <string>

const std::string dirPath = "test/api/cli/scriptargs/";

TEST_CASE("Print single script argument", TAG_SCRIPTARGS) {
    const std::string scriptPath = dirPath + "printSingleArg.daphne";
    compareDaphneToStr("123\n"        , scriptPath.c_str(), "--args", "foo=123");
    compareDaphneToStr("-123.45\n"    , scriptPath.c_str(), "--args", "foo=-123.45");
    compareDaphneToStr("1\n"          , scriptPath.c_str(), "--args", "foo=true");
    compareDaphneToStr("hello world\n", scriptPath.c_str(), "--args", "foo=\"hello world\"");
}

TEST_CASE("Missing script argument", TAG_SCRIPTARGS) {
    const std::string scriptPath = dirPath + "printSingleArg.daphne";
    // No script arguments provided.
    checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptPath.c_str());
    // Script arguments provided, but not the required one.
    checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptPath.c_str(), "--args", "bar=123");
}

TEST_CASE("Superfluous script argument", TAG_SCRIPTARGS) {
    const std::string scriptPath = dirPath + "printSingleArg.daphne";
    // TODO We could discuss if this is the desired behavior, since passing an
    // argument that is not used by the script might also indicate a mistake by
    // the user.
    // Passing superfluous script arguments, that are not actually used by the
    // script is okay.
    checkDaphneStatusCode(StatusCode::SUCCESS, scriptPath.c_str(), "--args", "foo=123,bar=456");
}

TEST_CASE("Duplicate script argument") {
    const std::string scriptPath = dirPath + "printSingleArg.daphne";
    checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptPath.c_str(), "--args", "foo=123,foo=456");
}

TEST_CASE("Ways of specifying script arguments", TAG_SCRIPTARGS) {
    std::stringstream out;
    std::stringstream err;
    
    const std::string scriptPath = dirPath + "printMultipleArgs.daphne";

    int status;
    SECTION("only after script file") {
        status = runDaphne(
                out, err,
                scriptPath.c_str(), "a=1", "b=2", "c=3", "d=4"
        );
    }
    SECTION("only via --args") {
        status = runDaphne(
                out, err,
                "--args", "a=1,b=2,c=3,d=4", scriptPath.c_str()
        );
    }
    SECTION("mixed") {
        status = runDaphne(
                out, err,
                "--args", "a=1,b=2", scriptPath.c_str(), "c=3", "d=4"
        );
    }
    
    // Don't REQUIRE, such that out and err are also printed in case of a test
    // failure. Don't use empty() on err, such that err is printed on failure.
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == "1\n2\n3\n4\n");
    CHECK(err.str() == "");
}