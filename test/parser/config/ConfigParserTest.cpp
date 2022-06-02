/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
 
#include <tags.h>

#include <catch.hpp>

#include <parser/config/ConfigParser.h>

#include <iostream>

const std::string dirPath = "test/parser/config/configFiles/";

TEST_CASE("Proper config file from src/api/cli directory")
{
    const std::string configFile = "src/api/cli/UserConfig.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_NOTHROW(ConfigParser::readUserConfig(configFile, userConfig));
}


TEST_CASE("Missing config file", TAG_PARSER)
{
    const std::string configFile = "-";
    REQUIRE_THROWS(ConfigParser::fileExists(configFile));
}

TEST_CASE("Empty config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig1.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_NOTHROW(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("Wrong JSON format in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig2.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("Wrong format of config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig3.txt";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}


TEST_CASE("Config file that contains only some keys", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig4.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_NOTHROW(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("Unknown key in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig5.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("The unknown value of param in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig6.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("An adequate enum value set in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig7.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_NOTHROW(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("An unknown enum value set in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig8.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}

TEST_CASE("Integer set as enum value instead of the name of the enum value in string format in the config file", TAG_PARSER)
{
    const std::string configFile = dirPath + "UserConfig9.json";
    DaphneUserConfig userConfig{};
    REQUIRE(ConfigParser::fileExists(configFile));
    REQUIRE_THROWS(ConfigParser::readUserConfig(configFile, userConfig));
}