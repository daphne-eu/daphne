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
// #include "build/antlr4_generated_src/DaphneSQLGrammar/DaphneSQLGrammarLexer.h"
// #include "build/antlr4_generated_src/DaphneSQLGrammar/DaphneSQLGrammarParser.h"
#include "DaphneSQLGrammarLexer.h"
#include "DaphneSQLGrammarParser.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

#include <istream>



void DaphneSQLParser::setView(std::unordered_map <std::string, mlir::Value> arg){
    view = arg;
}

void DaphneSQLParser::parseStream(mlir::OpBuilder & builder, std::istream & stream){
    //}, std::unordered_map<std::string, mlir::Value>& tables) {
    mlir::Location loc = builder.getUnknownLoc();
    // Create a single "main"-function and insert DaphneIR operations into it.
    // auto * funcBlock = new mlir::Block();
    {
        // mlir::OpBuilder::InsertionGuard guard(builder);
        // builder.setInsertionPoint(funcBlock, funcBlock->begin());

        // Run ANTLR-based DaphneSQL parser.
        antlr4::ANTLRInputStream input(stream);
        input.name = "whateverFile"; // TODO
        DaphneSQLGrammarLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        DaphneSQLGrammarParser parser(&tokens);
        DaphneSQLGrammarParser::SqlContext * ctx = parser.sql();
        DaphneSQLVisitor visitor(builder, view);
        visitor.visitSql(ctx);

    }
    // auto * terminator = funcBlock->getTerminator();
    // auto funcType = mlir::FunctionType::get(
    //         builder.getContext(),
    //         funcBlock->getArgumentTypes(),
    //         terminator->getOperandTypes()
    // );
    // auto func = builder.create<mlir::FuncOp>(loc, "sql", funcType);
    // func.push_back(funcBlock);
}
