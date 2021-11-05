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
#include <parser/daphnedsl/DaphneDSLParser.h>
#include <parser/daphnedsl/DaphneDSLVisitor.h>
#include <parser/CancelingErrorListener.h>

#include "antlr4-runtime.h"
#include "DaphneDSLGrammarLexer.h"
#include "DaphneDSLGrammarParser.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

#include <istream>

void DaphneDSLParser::parseStream(mlir::OpBuilder & builder, std::istream & stream, const std::string &sourceName) {
    CancelingErrorListener errorListener;
    auto errorStrategy = std::make_shared<antlr4::BailErrorStrategy>();
    mlir::Location loc = mlir::FileLineColLoc::get(builder.getIdentifier(sourceName), 0, 0);
    
    // Create a single "main"-function and insert DaphneIR operations into it.
    auto * funcBlock = new mlir::Block();
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(funcBlock, funcBlock->begin());
        
        // Run ANTLR-based DaphneDSL parser.
        antlr4::ANTLRInputStream input(stream);
        input.name = sourceName;
        DaphneDSLGrammarLexer lexer(&input);
        lexer.removeErrorListeners();
        lexer.addErrorListener(&errorListener);
        antlr4::CommonTokenStream tokens(&lexer);
        DaphneDSLGrammarParser parser(&tokens);
        // TODO: evaluate if overloading error handler makes sense
        parser.setErrorHandler(errorStrategy);
        DaphneDSLGrammarParser::ScriptContext * ctx = parser.script();
        DaphneDSLVisitor visitor(module, builder, args);
        visitor.visitScript(ctx);
        
        builder.create<mlir::daphne::ReturnOp>(loc);
    }
    auto * terminator = funcBlock->getTerminator();
    auto funcType = builder.getFunctionType(
        funcBlock->getArgumentTypes(),
        terminator->getOperandTypes()
    );
    auto func = builder.create<mlir::FuncOp>(loc, "main", funcType);
    func.push_back(funcBlock);
}
