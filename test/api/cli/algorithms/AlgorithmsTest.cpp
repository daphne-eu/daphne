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

#include <sstream>
#include <string>

const std::string dirPath = "test/api/cli/algorithms/";

// For now, we just check if the algorithms terminate successfully, but we
// don't check the results.

TEST_CASE("components", TAG_ALGORITHMS) {
    checkDaphneStatusCode(
        StatusCode::SUCCESS, dirPath + "components.daphne",
        "--args", "n=100", "e=1000"
    );
}

TEST_CASE("componentsSparse", TAG_ALGORITHMS) {
    // TODO: check against mode without `--select-matrix-representations` by reading file or when the sparse and dense
    //  random kernels have the same values for same seed
    checkDaphneStatusCode(
        StatusCode::SUCCESS, dirPath + "components.daphne",
        "--select-matrix-representations", "--args", "n=100", "e=100"
    );
}

TEST_CASE("kmeans", TAG_ALGORITHMS) {
    checkDaphneStatusCode(
            StatusCode::SUCCESS, dirPath + "kmeans.daphne",
            "--args", "r=100", "c=5", "f=20", "i=10"
    );
}

TEST_CASE("lm", TAG_ALGORITHMS) {
    checkDaphneStatusCode(
            StatusCode::SUCCESS, dirPath + "lm.daphne",
            "--args", "r=100", "c=20"
    );
}