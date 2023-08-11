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
#include <parser/sql/SQLParser.h>
#include <parser/sql/SQLVisitor.h>

#include "antlr4-runtime.h"
#include "SQLGrammarLexer.h"
#include "SQLGrammarParser.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

#include <spdlog/spdlog.h>

#include <istream>
#include <parser/CancelingErrorListener.h>

void SQLParser::setView(std::unordered_map <std::string, mlir::Value> arg){
    view = arg;
}

mlir::Value SQLParser::parseStreamFrame(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName){
    CancelingErrorListener errorListener;
    auto errorStrategy = std::make_shared<antlr4::BailErrorStrategy>();
    {
        antlr4::ANTLRInputStream input(stream);
        input.name = sourceName;
        SQLGrammarLexer lexer(&input);
        lexer.removeErrorListeners();
        lexer.addErrorListener(&errorListener);
        antlr4::CommonTokenStream tokens(&lexer);
        SQLGrammarParser parser(&tokens);
        // TODO: evaluate if overloading error handler makes sense
        parser.setErrorHandler(errorStrategy);
        SQLGrammarParser::SqlContext * ctx = parser.sql();
        SQLVisitor visitor(builder, view);
        antlrcpp::Any a;
        try {
            a = visitor.visitSql(ctx);
        }
        catch (std::runtime_error& re) {
            spdlog::error("Caught std::runtime_error in {}:{}: \n{}",__FILE__, __LINE__, re.what());
            throw;
        }

        if(a.is<mlir::Value>()){
          return a.as<mlir::Value>();
        }
        throw std::runtime_error("expected a mlir::Value");
    }
}

void SQLParser::parseStream(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName){
    parseStreamFrame(builder, stream, sourceName);
}
