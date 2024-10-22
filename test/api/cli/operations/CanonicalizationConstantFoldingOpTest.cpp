/*
 * Copyright 2023 The DAPHNE Consortium
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
#include <string>


const std::string dirPath = "test/api/cli/operations/";

void compareDaphneParsingSimplifiedToRef(const std::string & refFilePath, const std::string & scriptFilePath) {
    std::stringstream out;
    std::stringstream err;
	const std::string exp = readTextFile(refFilePath);
    int status = runDaphne(out, err, "--explain=parsing_simplified", scriptFilePath.c_str());
    CHECK(status == StatusCode::SUCCESS);
    CHECK(err.str() == exp);
}

TEST_CASE("additive_inverse_constant_folding", TAG_CODEGEN TAG_OPERATIONS) {
    const std::string testName = "addinv_constant_folding";
    compareDaphneParsingSimplifiedToRef(dirPath + testName + ".txt", dirPath + testName + ".daphne");
}

TEST_CASE("additive_inverse_canonicalization", TAG_CODEGEN TAG_OPERATIONS) {
    const std::string testName = "addinv_canonicalization";
    compareDaphneParsingSimplifiedToRef(dirPath + testName + ".txt", dirPath + testName + ".daphne");
}

TEST_CASE("binary_operator_casts_constant_folding", TAG_CODEGEN TAG_OPERATIONS) {
    const std::string testName = "binary_op_casts_constant_folding";
    compareDaphneParsingSimplifiedToRef(dirPath + testName + ".txt", dirPath + testName + ".daphne");
}
