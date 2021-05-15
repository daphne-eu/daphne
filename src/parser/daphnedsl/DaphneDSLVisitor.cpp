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
#include <parser/daphnedsl/DaphneDSLVisitor.h>
#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
#include "DaphneDSLGrammarParser.h"

#include <stdexcept>
#include <string>

#include <cstdint>
#include <cstdlib>

// ****************************************************************************
// Helper functions
// ****************************************************************************

mlir::Value valueOrError(antlrcpp::Any a) {
    if(a.is<mlir::Value>())
        return a.as<mlir::Value>();
    throw std::runtime_error("something was expected to be an mlir::Value, but it was none");
}

// ****************************************************************************
// Visitor functions
// ****************************************************************************

antlrcpp::Any DaphneDSLVisitor::visitScript(DaphneDSLGrammarParser::ScriptContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitStatement(DaphneDSLGrammarParser::StatementContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitBlockStatement(DaphneDSLGrammarParser::BlockStatementContext * ctx) {
    symbolTable.pushScope();
    antlrcpp::Any res = visitChildren(ctx);
    symbolTable.put(symbolTable.popScope());
    return res;
}

antlrcpp::Any DaphneDSLVisitor::visitExprStatement(DaphneDSLGrammarParser::ExprStatementContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitAssignStatement(DaphneDSLGrammarParser::AssignStatementContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->expr())));
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitLiteralExpr(DaphneDSLGrammarParser::LiteralExprContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitIdentifierExpr(DaphneDSLGrammarParser::IdentifierExprContext * ctx) {
    std::string var = ctx->var->getText();
    try {
        return symbolTable.get(var);
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("variable " + var + " referenced before assignment");
    }
}

antlrcpp::Any DaphneDSLVisitor::visitParanthesesExpr(DaphneDSLGrammarParser::ParanthesesExprContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitCallExpr(DaphneDSLGrammarParser::CallExprContext * ctx) {
    // TODO Reduce the code duplication here.
    std::string func = ctx->func->getText();
    const size_t numArgs = ctx->expr().size();
    mlir::Location loc = builder.getUnknownLoc();
    if(func == "print") {
        if(numArgs != 1)
            throw std::runtime_error("function print expects exactly 1 argument(s)");
        return builder.create<mlir::daphne::PrintOp>(
                loc, valueOrError(visit(ctx->expr(0)))
        );
    }
    if(func == "rand") {
        if(numArgs != 6)
            throw std::runtime_error("function rand expects exactly 6 argument(s)");
        mlir::Value numRows = valueOrError(visit(ctx->expr(0)));
        mlir::Value numCols = valueOrError(visit(ctx->expr(1)));
        mlir::Value min = valueOrError(visit(ctx->expr(2)));
        mlir::Value max = valueOrError(visit(ctx->expr(3)));
        mlir::Value sparsity = valueOrError(visit(ctx->expr(4)));
        mlir::Value seed = valueOrError(visit(ctx->expr(5)));
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::RandMatrixOp>(
                        loc,
                        mlir::daphne::MatrixType::get(builder.getContext(), min.getType()),
                        numRows, numCols, min, max, sparsity, seed
                )
        );
    }
    if(func == "t") {
        if(numArgs != 1)
            throw std::runtime_error("function t expects exactly 1 argument(s)");
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::TransposeOp>(
                        loc, valueOrError(visit(ctx->expr(0)))
                )
        );
    }
    throw std::runtime_error("unexpected func");
}

antlrcpp::Any DaphneDSLVisitor::visitMatmulExpr(DaphneDSLGrammarParser::MatmulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = valueOrError(visit(ctx->lhs));
    mlir::Value rhs = valueOrError(visit(ctx->rhs));
    
    if(op == "@")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MatMulOp>(loc, lhs.getType(), lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMulExpr(DaphneDSLGrammarParser::MulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = valueOrError(visit(ctx->lhs));
    mlir::Value rhs = valueOrError(visit(ctx->rhs));
    
    if(op == "*")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwMulOp>(loc, lhs, rhs));
    if(op == "/")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwDivOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitAddExpr(DaphneDSLGrammarParser::AddExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = valueOrError(visit(ctx->lhs));
    mlir::Value rhs = valueOrError(visit(ctx->rhs));
    
    if(op == "+")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAddOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCmpExpr(DaphneDSLGrammarParser::CmpExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = valueOrError(visit(ctx->lhs));
    mlir::Value rhs = valueOrError(visit(ctx->rhs));
    
    if(op == "==")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwEqOp>(loc, lhs, rhs));
    if(op == "!=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwNeqOp>(loc, lhs, rhs));
    if(op == "<")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwLtOp>(loc, lhs, rhs));
    if(op == "<=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwLeOp>(loc, lhs, rhs));
    if(op == ">")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwGtOp>(loc, lhs, rhs));
    if(op == ">=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwGeOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitLiteral(DaphneDSLGrammarParser::LiteralContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    if(auto lit = ctx->INT_LITERAL()) {
        int64_t val = atol(lit->getText().c_str());
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                        loc,
                        builder.getIntegerAttr(builder.getIntegerType(64, true), val)
                )
        );
    }
    if(auto lit = ctx->FLOAT_LITERAL()) {
        double val = atof(lit->getText().c_str());
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                        loc,
                        builder.getF64FloatAttr(val)
                )
        );
    }
    throw std::runtime_error("unexpected literal");
}