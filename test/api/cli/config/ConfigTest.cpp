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

TEST_CASE("config success", TAG_CONFIG) {
    // Providing the following example config files as --config must succeed.
    for(const char * configFilePath : {
        "src/api/cli/UserConfig.json",
                
        "test/parser/config/configFiles/UserConfig1.json",
        "test/parser/config/configFiles/UserConfig4.json",
        "test/parser/config/configFiles/UserConfig7.json"
    }) {
        DYNAMIC_SECTION(configFilePath) {
            checkDaphneStatusCode(
                    StatusCode::SUCCESS,
                    "test/api/cli/config/empty.daphne", "--config", configFilePath
            );
        }
    }
}

TEST_CASE("config failure", TAG_CONFIG) {
    // Providing the following example config files as --config must fail.
    for(const char * configFilePath : {
        "test/parser/config/configFiles/UserConfig2.json",
        "test/parser/config/configFiles/UserConfig3.txt",
        "test/parser/config/configFiles/UserConfig5.json",
        "test/parser/config/configFiles/UserConfig6.json",
        "test/parser/config/configFiles/UserConfig8.json",
        "test/parser/config/configFiles/UserConfig9.json"
    }) {
        DYNAMIC_SECTION(configFilePath) {
            checkDaphneStatusCode(
                    StatusCode::PARSER_ERROR,
                    "test/api/cli/config/empty.daphne", "--config", configFilePath
            );
        }
    }
}