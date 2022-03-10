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

#ifndef SRC_PARSER_CONFIG_CONFIGPARSER_H
#define SRC_PARSER_CONFIG_CONFIGPARSER_H

#include "../../../thirdparty/nlohmannjson/json.hpp"
#include "../../api/cli/DaphneUserConfig.h"
#include <string>

class ConfigParser {
public:
    static bool fileExists(const std::string& filename);
    static void readUserConfig(const std::string& filename, DaphneUserConfig& config);
private:
    static bool keyExists(const nlohmann::json& j, const std::string& key);
    static void checkAnyUnexpectedKeys(const nlohmann::basic_json<>& j, const std::string& filename);
};

#endif