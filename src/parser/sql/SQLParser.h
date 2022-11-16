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

#ifndef SRC_PARSER_SQL_SQLPARSER_H
#define SRC_PARSER_SQL_SQLPARSER_H

#include <parser/Parser.h>

#include <mlir/IR/Builders.h>

#include <istream>
#include <string>
#include <unordered_map>

/**
 * @brief DAPHNEs custom Parser dedicated for parsing SQL strings.
 *
 * Implements the daphne::Parser interface.
 */
class SQLParser : public Parser {
  public:
    using viewType = std::unordered_map <std::string, mlir::Value>;
  private:
    viewType view;
  
  public:
    /**
     * Set the view on Frames (name to Frame mapping) used while emitting DaphneIR from parsed ANTLR tree.
     * @param view unordered map holding name to Frame mapping
     */
    void setView(viewType view);
    void parseStream(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName) override;
    mlir::Value parseStreamFrame(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName) const;
};

#endif /* SRC_PARSER_SQL_SQLPARSER_H */
