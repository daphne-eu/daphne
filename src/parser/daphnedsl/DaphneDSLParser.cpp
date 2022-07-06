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
    // TODO: we could remove `sourceName` arg and instead use location from module for filename
    auto module = llvm::cast<mlir::ModuleOp>(builder.getBlock()->getParentOp());

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
        parser.removeErrorListeners();
        parser.addErrorListener(&errorListener);

        DaphneDSLGrammarParser::ScriptContext * ctx = parser.script();
        DaphneDSLVisitor visitor(module, builder, args, sourceName, userConf);
        visitor.visitScript(ctx);

        mlir::Location loc = mlir::FileLineColLoc::get(builder.getIdentifier(sourceName), 0, 0);
        if(!builder.getBlock()->empty()) {
            loc = builder.getBlock()->back().getLoc();
        }
        builder.create<mlir::daphne::ReturnOp>(loc);
    }
    auto * terminator = funcBlock->getTerminator();
    auto funcType = builder.getFunctionType(
        funcBlock->getArgumentTypes(),
        terminator->getOperandTypes()
    );
    auto loc = mlir::FileLineColLoc::get(builder.getIdentifier(sourceName), 0, 0);
    auto func = builder.create<mlir::FuncOp>(loc, "main", funcType);
    func.push_back(funcBlock);
}
