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

#include <ir/daphneir/Daphne.h>
#include <parser/daphnesql/DaphneSQLParser.h>
#include <parser/daphnesql/DaphneSQLVisitor.h>

#include "antlr4-runtime.h"
#include "DaphneSQLGrammarLexer.h"
#include "DaphneSQLGrammarParser.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

#include <istream>

void DaphneSQLParser::setView(std::unordered_map <std::string, mlir::Value> arg){
    view = arg;
}

mlir::Value DaphneSQLParser::parseStreamFrame(mlir::OpBuilder & builder, std::istream & stream){
    mlir::Location loc = builder.getUnknownLoc();
    {
        antlr4::ANTLRInputStream input(stream);
        input.name = "whateverFile"; // TODO
        DaphneSQLGrammarLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        DaphneSQLGrammarParser parser(&tokens);
        DaphneSQLGrammarParser::SqlContext * ctx = parser.sql();
        DaphneSQLVisitor visitor(builder, view);
        antlrcpp::Any a = visitor.visitSql(ctx);
        if(a.is<mlir::Value>()){
          return a.as<mlir::Value>();
        }
        throw std::runtime_error("expected a mlir::Value");
    }
}

void DaphneSQLParser::parseStream(mlir::OpBuilder & builder, std::istream & stream){
    parseStreamFrame(builder, stream);
}
