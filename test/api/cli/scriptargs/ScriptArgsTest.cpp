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

const std::string scriptPath = "test/api/cli/scriptargs/printSingleArg.daphne";

TEST_CASE("Print single script argument", TAG_SCRIPTARGS) {
    compareDaphneToStr("123\n"        , scriptPath, "--args", "foo=123");
    compareDaphneToStr("-123.45\n"    , scriptPath, "--args", "foo=-123.45");
    compareDaphneToStr("1\n"          , scriptPath, "--args", "foo=true");
    compareDaphneToStr("hello world\n", scriptPath, "--args", "foo=\"hello world\"");
}

TEST_CASE("Missing script argument", TAG_SCRIPTARGS) {
    // No script arguments provided.
    checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptPath);
    // Script arguments provided, but not the required one.
    checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptPath, "--args", "bar=123");
}

TEST_CASE("Superfluous script argument", TAG_SCRIPTARGS) {
    // TODO We could discuss if this is the desired behavior, since passing an
    // argument that is not used by the script might also indicate a mistake by
    // the user.
    // Passing superfluous script arguments, that are not actually used by the
    // script is okay.
    checkDaphneStatusCode(StatusCode::SUCCESS, scriptPath, "--args", "foo=123,bar=456");
}