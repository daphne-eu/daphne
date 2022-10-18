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

#ifndef DAPHNE_PROTOTYPE_MORPHSTORESQLPARSER_H
#define DAPHNE_PROTOTYPE_MORPHSTORESQLPARSER_H

#include <parser/Parser.h>

#include <mlir/IR/Builders.h>

#include <istream>
#include <string>
#include <unordered_map>

struct MorphStoreSQLParser : public Parser{

    std::unordered_map <std::string, mlir::Value> view;
    void setView(std::unordered_map <std::string, mlir::Value> view);

    void parseStream(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName) override;

    mlir::Value parseStreamFrame(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName);

};


#endif //DAPHNE_PROTOTYPE_MORPHSTORESQLPARSER_H
