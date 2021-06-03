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

#include <mlir/Dialect/SCF/SCF.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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

mlir::Value DaphneDSLVisitor::castIf(mlir::Location loc, mlir::Type t, mlir::Value v) {
    if(v.getType() == t)
        return v;
    return builder.create<mlir::daphne::CastOp>(loc, t, v);
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
    std::string var = ctx->var->getText();
    if(symbolTable.has(var) && symbolTable.get(var).isReadOnly)
        throw std::runtime_error("trying to assign read-only variable " + var);
    symbolTable.put(var, ScopedSymbolTable::SymbolInfo(valueOrError(visit(ctx->expr())), false));
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitIfStatement(DaphneDSLGrammarParser::IfStatementContext * ctx) {
    mlir::Value cond = valueOrError(visit(ctx->cond));
    
    mlir::Location loc = builder.getUnknownLoc();

    // Save the current state of the builder.
    mlir::OpBuilder oldBuilder = builder;
    
    // Generate the operations for the then-block.
    mlir::Block thenBlock;
    builder.setInsertionPointToEnd(&thenBlock);
    symbolTable.pushScope();
    visit(ctx->thenStmt);
    ScopedSymbolTable::SymbolTable owThen = symbolTable.popScope();
    
    // Generate the operations for the else-block, if it is present. Otherwise
    // leave it empty; we might need to insert a yield-operation.
    mlir::Block elseBlock;
    ScopedSymbolTable::SymbolTable owElse;
    if(ctx->elseStmt) {
        builder.setInsertionPointToEnd(&elseBlock);
        symbolTable.pushScope();
        visit(ctx->elseStmt);
        owElse = symbolTable.popScope();
    }
    
    // Determine the result type(s) of the if-operation as well as the operands
    // to the yield-operation of both branches.
    std::set<std::string> owUnion = ScopedSymbolTable::mergeSymbols(owThen, owElse);
    std::vector<mlir::Type> resultTypes;
    std::vector<mlir::Value> resultsThen;
    std::vector<mlir::Value> resultsElse;
    for(auto it = owUnion.begin(); it != owUnion.end(); it++) {
        mlir::Value valThen = symbolTable.get(*it, owThen).value;
        mlir::Value valElse = symbolTable.get(*it, owElse).value;
        if(valThen.getType() != valElse.getType())
            // TODO We could try to cast the types.
            throw std::runtime_error("type missmatch");
        resultTypes.push_back(valThen.getType());
        resultsThen.push_back(valThen);
        resultsElse.push_back(valElse);
    }

    // Create yield-operations in both branches, possibly with empty results.
    builder.setInsertionPointToEnd(&thenBlock);
    builder.create<mlir::scf::YieldOp>(loc, resultsThen);
    builder.setInsertionPointToEnd(&elseBlock);
    builder.create<mlir::scf::YieldOp>(loc, resultsElse);
    
    // Restore the old state of the builder.
    builder = oldBuilder;
    
    // Helper functions to move the operations in the two blocks created above
    // into the actual branches of the if-operation.
    auto insertThenBlockDo = [&](mlir::OpBuilder & nested, mlir::Location loc) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), thenBlock.getOperations());
    };
    auto insertElseBlockDo = [&](mlir::OpBuilder & nested, mlir::Location loc) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), elseBlock.getOperations());
    };
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> insertElseBlockNo = nullptr;
    
    // Create the actual if-operation. Generate the else-block only if it was
    // explicitly given in the DSL script, or when it is needed to yield values.
    auto ifOp = builder.create<mlir::scf::IfOp>(
            loc,
            resultTypes,
            cond,
            insertThenBlockDo,
            (ctx->elseStmt || !owUnion.empty()) ? insertElseBlockDo : insertElseBlockNo
    );
    
    // Rewire the results of the if-operation to their variable names.
    size_t i = 0;
    for(auto it = owUnion.begin(); it != owUnion.end(); it++)
        symbolTable.put(*it, ifOp.results()[i++]);
    
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitWhileStatement(DaphneDSLGrammarParser::WhileStatementContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    
    auto ip = builder.saveInsertionPoint();
    
    // The two blocks for the SCF WhileOp.
    auto beforeBlock = new mlir::Block;
    auto afterBlock = new mlir::Block;
    
    const bool isDoWhile = ctx->KW_DO();
    
    mlir::Value cond;
    ScopedSymbolTable::SymbolTable ow;
    if(isDoWhile) { // It's a do-while loop.
        builder.setInsertionPointToEnd(beforeBlock);
        
        // Scope for body and condition, such that condition can see the body's
        // updates to variables existing before the loop.
        symbolTable.pushScope();
        
        // The body gets its own scope to not expose variables created inside
        // the body to the condition. While this is unnecessary if the body is
        // a block statement, there are nasty cases if no block statement is
        // used.
        symbolTable.pushScope();
        visit(ctx->bodyStmt);
        ow = symbolTable.popScope();
        
        // Make the body's updates visible to the condition.
        symbolTable.put(ow);
        
        cond = valueOrError(visit(ctx->cond));
        
        symbolTable.popScope();
    }
    else { // It's a while loop.
        builder.setInsertionPointToEnd(beforeBlock);
        cond = valueOrError(visit(ctx->cond));

        builder.setInsertionPointToEnd(afterBlock);
        symbolTable.pushScope();
        visit(ctx->bodyStmt);
        ow = symbolTable.popScope();
    }
    
    // Determine which variables created before the loop are updated in the
    // loop's body. These become the arguments and results of the WhileOp and
    // its "before" and "after" region.
    std::vector<mlir::Value> owVals;
    std::vector<mlir::Type> resultTypes;
    std::vector<mlir::Value> whileOperands;
    for(auto it = ow.begin(); it != ow.end(); it++) {
        mlir::Value owVal = it->second.value;
        mlir::Type type = owVal.getType();
        
        owVals.push_back(owVal);
        resultTypes.push_back(type);
        
        mlir::Value oldVal = symbolTable.get(it->first).value;
        whileOperands.push_back(oldVal);
        
        beforeBlock->addArgument(type);
        afterBlock->addArgument(type);
    }
    
    // Create the ConditionOp of the "before" block.
    builder.setInsertionPointToEnd(beforeBlock);
    if(isDoWhile)
        builder.create<mlir::scf::ConditionOp>(loc, cond, owVals);
    else
        builder.create<mlir::scf::ConditionOp>(loc, cond, beforeBlock->getArguments());
    
    // Create the YieldOp of the "after" block.
    builder.setInsertionPointToEnd(afterBlock);
    if(isDoWhile)
        builder.create<mlir::scf::YieldOp>(loc, afterBlock->getArguments());
    else
        builder.create<mlir::scf::YieldOp>(loc, owVals);
    
    builder.restoreInsertionPoint(ip);
    
    // Create the SCF WhileOp and insert the "before" and "after" blocks.
    auto whileOp = builder.create<mlir::scf::WhileOp>(loc, resultTypes, whileOperands);
    whileOp.before().push_back(beforeBlock);
    whileOp.after().push_back(afterBlock);
    
    size_t i = 0;
    for(auto it = ow.begin(); it != ow.end(); it++) {
        // Replace usages of the variables updated in the loop's body by the
        // corresponding block arguments.
        whileOperands[i].replaceUsesWithIf(beforeBlock->getArgument(i), [&](mlir::OpOperand & operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.before().isAncestor(parentRegion);
        });
        whileOperands[i].replaceUsesWithIf(afterBlock->getArgument(i), [&](mlir::OpOperand & operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.after().isAncestor(parentRegion);
        });
        
        // Rewire the results of the WhileOp to their variable names.
        symbolTable.put(it->first, whileOp.results()[i++]);
    }
    
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitForStatement(DaphneDSLGrammarParser::ForStatementContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    
    // The type we assume for from, to, and step.
    mlir::Type t = builder.getIntegerType(64, true);
    
    // Parse from, to, and step.
    mlir::Value from = castIf(loc, t, valueOrError(visit(ctx->from)));
    mlir::Value to = castIf(loc, t, valueOrError(visit(ctx->to)));
    mlir::Value step;
    mlir::Value direction; // count upwards (+1) or downwards (-1)
    if(ctx->step) {
        // If the step is given, parse it and derive the counting direction.
        step = castIf(loc, t, valueOrError(visit(ctx->step)));
        direction = builder.create<mlir::daphne::EwSignOp>(loc, t, step);
    }
    else {
        // If the step is not given, derive it as `-1 + 2 * (to >= from)`,
        // which always results in -1 or +1, even if to equals from.
        step = builder.create<mlir::daphne::EwAddOp>(
                loc,
                builder.create<mlir::daphne::ConstantOp>(loc, builder.getIntegerAttr(t, -1)),
                builder.create<mlir::daphne::EwMulOp>(
                        loc,
                        builder.create<mlir::daphne::ConstantOp>(loc, builder.getIntegerAttr(t, 2)),
                        castIf(loc, t, builder.create<mlir::daphne::EwGeOp>(loc, to, from))
                )
        );
        direction = step;
    }
    // Compensate for the fact that the upper bound of SCF's ForOp is exclusive,
    // while we want it to be inclusive.
    to = builder.create<mlir::daphne::EwAddOp>(loc, to, direction);
    // Compensate for the fact that SCF's ForOp can only count upwards.
    from = builder.create<mlir::daphne::EwMulOp>(loc, from, direction);
    to   = builder.create<mlir::daphne::EwMulOp>(loc, to  , direction);
    step = builder.create<mlir::daphne::EwMulOp>(loc, step, direction);
    // Compensate for the fact that SCF's ForOp expects its parameters to be of
    // MLIR's IndexType.
    mlir::Type idxType = builder.getIndexType();
    from = castIf(loc, idxType, from);
    to   = castIf(loc, idxType, to);
    step = castIf(loc, idxType, step);
    
    auto ip = builder.saveInsertionPoint();

    // A block for the body of the for-loop.
    mlir::Block bodyBlock;
    builder.setInsertionPointToEnd(&bodyBlock);
    symbolTable.pushScope();
    
    // A placeholder for the loop's induction variable, since we do not know it
    // yet; will be replaced later.
    mlir::Value ph = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIndexAttr(123));
    // Make the induction variable available by the specified name.
    symbolTable.put(
            ctx->var->getText(),
            ScopedSymbolTable::SymbolInfo(
                    // Un-compensate for counting direction.
                    builder.create<mlir::daphne::EwMulOp>(
                            loc, castIf(loc, t, ph), direction
                    ),
                    true // the for-loop's induction variable is read-only
            )
    );

    // Parse the loop's body.
    visit(ctx->bodyStmt);
    
    // Determine which variables created before the loop are updated in the
    // loop's body. These become the arguments and results of the ForOp.
    ScopedSymbolTable::SymbolTable ow = symbolTable.popScope();
    std::vector<mlir::Value> resVals;
    std::vector<mlir::Value> forOperands;
    for(auto it = ow.begin(); it != ow.end(); it++) {
        resVals.push_back(it->second.value);
        forOperands.push_back(symbolTable.get(it->first).value);
    }
    
    builder.create<mlir::scf::YieldOp>(loc, resVals);
    
    builder.restoreInsertionPoint(ip);
    
    // Helper function for moving the operations in the block created above
    // into the actual body of the ForOp.
    auto insertBodyBlock = [&](mlir::OpBuilder & nested, mlir::Location loc, mlir::Value iv, mlir::ValueRange lcv) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), bodyBlock.getOperations());
    };
    
    // Create the actual ForOp.
    auto forOp = builder.create<mlir::scf::ForOp>(loc, from, to, step, forOperands, insertBodyBlock);

    // Substitute the induction variable, now that we know it.
    ph.replaceAllUsesWith(forOp.getInductionVar());
    
    size_t i = 0;
    for(auto it = ow.begin(); it != ow.end(); it++) {
        // Replace usages of the variables updated in the loop's body by the
        // corresponding block arguments.
        forOperands[i].replaceUsesWithIf(forOp.getRegionIterArgs()[i], [&](mlir::OpOperand & operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && forOp.getLoopBody().isAncestor(parentRegion);
        });
        
        // Rewire the results of the ForOp to their variable names.
        symbolTable.put(it->first, forOp.results()[i]);
        
        i++;
    }
    
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitLiteralExpr(DaphneDSLGrammarParser::LiteralExprContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitIdentifierExpr(DaphneDSLGrammarParser::IdentifierExprContext * ctx) {
    std::string var = ctx->var->getText();
    try {
        return symbolTable.get(var).value;
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
    mlir::Type sizeType = builder.getIndexType();
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
        mlir::Value numRows = castIf(loc, sizeType, valueOrError(visit(ctx->expr(0))));
        mlir::Value numCols = castIf(loc, sizeType, valueOrError(visit(ctx->expr(1))));
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