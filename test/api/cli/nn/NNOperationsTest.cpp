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
#include "api/cli/StatusCode.h"

const std::string dirPath = "test/api/cli/nn/";

const std::string expected_output = "diff: 0\n";

/*
 * 1...dsl forward
 * 2...dsl backward
 * 3...kernel forward
 * 4...kernel backward
 */

//                    const std::string expected_result = std::string("expected=\"") + dirPath + base_name + std::string(".csv\"");
//                    const std::string args = "input=\"test/data/mnist20/mnist20_features.csv\",eps=1e-9f," + expected_result;
#define MAKE_TEST_CASE(name, count) \
    TEST_CASE(name, TAG_DNN) { \
        for(unsigned i = 1; i <= count; i++) { \
            DYNAMIC_SECTION(name "_" << i << ".daph") { \
                    const std::string base_name = name + std::string("_") + std::to_string(i); \
                    const std::string script_name =  base_name + std::string(".daph"); \
                    compareDaphneToStr(expected_output, dirPath + script_name, "--config", "test/api/cli/nn/UserConfig.json"); \
            } \
        } \
    }

MAKE_TEST_CASE("activation_relu_dsl", 2)
MAKE_TEST_CASE("conv2d_dsl", 2)
MAKE_TEST_CASE("conv2d_kernel", 2)

/*
MAKE_TEST_CASE("affine_dsl", 1)
MAKE_TEST_CASE("batchnorm2d_dsl", 1)
MAKE_TEST_CASE("dropout_dsl", 1)
MAKE_TEST_CASE("loss_dsl", 1)
MAKE_TEST_CASE("pooling", 1)
MAKE_TEST_CASE("optimizer_dsl", 1)
MAKE_TEST_CASE("softmax_dsl", 1)
 */