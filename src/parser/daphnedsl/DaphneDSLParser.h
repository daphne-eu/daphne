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

#pragma once

#include <parser/Parser.h>

#include <mlir/IR/Builders.h>
#include <api/cli/DaphneUserConfig.h>

#include <istream>
#include <string>
#include <unordered_map>
#include <utility>

class DaphneDSLParser : public Parser {

    std::unordered_map<std::string, std::string> args;
    DaphneUserConfig userConf;
    
public:

    DaphneDSLParser(std::unordered_map<std::string, std::string> args, DaphneUserConfig userConf) :
            args(std::move(args)), userConf(std::move(userConf)) { }

    DaphneDSLParser() : DaphneDSLParser(std::unordered_map<std::string, std::string>(), DaphneUserConfig()) { }

    void parseStream(mlir::OpBuilder &builder, std::istream &stream, const std::string &sourceName) override;
    
};
