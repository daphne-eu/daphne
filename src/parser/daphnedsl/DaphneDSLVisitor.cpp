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
#include "DaphneDSLGrammarLexer.h"
#include "DaphneDSLGrammarParser.h"

#include <mlir/Dialect/SCF/SCF.h>

#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>

// ****************************************************************************
// Utilities
// ****************************************************************************

void handleAssignmentPart(
        const std::string & var,
        ScopedSymbolTable & symbolTable,
        mlir::Value val
) {
    if(symbolTable.has(var) && symbolTable.get(var).isReadOnly)
        throw std::runtime_error("trying to assign read-only variable " + var);
    symbolTable.put(var, ScopedSymbolTable::SymbolInfo(val, false));
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
    const size_t numVars = ctx->IDENTIFIER().size();
    antlrcpp::Any rhsAny = visit(ctx->expr());
    bool rhsIsRR = rhsAny.is<mlir::ResultRange>();
    if(numVars == 1) {
        // A single variable on the left-hand side.
        if(rhsIsRR)
            throw std::runtime_error(
                    "trying to assign multiple results to a single variable"
            );
        handleAssignmentPart(
                ctx->IDENTIFIER(0)->getText(),
                symbolTable,
                utils.valueOrError(rhsAny)
        );
        return nullptr;
    }
    else if(numVars > 1) {
        // Multiple variables on the left-hand side; the expression must be an
        // operation returning multiple outputs.
        if(rhsIsRR) {
            auto rhsAsRR = rhsAny.as<mlir::ResultRange>();
            if(rhsAsRR.size() == numVars) {
                for(size_t i = 0; i < numVars; i++)
                    handleAssignmentPart(
                            ctx->IDENTIFIER(i)->getText(), symbolTable, rhsAsRR[i]
                    );
                return nullptr;
            }
        }
        throw std::runtime_error(
                "right-hand side expression of assignment to multiple "
                "variables must return multiple values, one for each "
                "variable on the left-hand side"
        );
    }
    assert(
            false && "the DaphneDSL grammar should prevent zero variables "
            "on the left-hand side of an assignment"
    );
}

antlrcpp::Any DaphneDSLVisitor::visitIfStatement(DaphneDSLGrammarParser::IfStatementContext * ctx) {
    mlir::Value cond = utils.castBoolIf(utils.valueOrError(visit(ctx->cond)));
    
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
        
        cond = utils.castBoolIf(utils.valueOrError(visit(ctx->cond)));
        
        symbolTable.popScope();
    }
    else { // It's a while loop.
        builder.setInsertionPointToEnd(beforeBlock);
        cond = utils.castBoolIf(utils.valueOrError(visit(ctx->cond)));

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
    mlir::Value from = utils.castIf(t, utils.valueOrError(visit(ctx->from)));
    mlir::Value to = utils.castIf(t, utils.valueOrError(visit(ctx->to)));
    mlir::Value step;
    mlir::Value direction; // count upwards (+1) or downwards (-1)
    if(ctx->step) {
        // If the step is given, parse it and derive the counting direction.
        step = utils.castIf(t, utils.valueOrError(visit(ctx->step)));
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
                        utils.castIf(t, builder.create<mlir::daphne::EwGeOp>(loc, to, from))
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
    from = utils.castIf(idxType, from);
    to   = utils.castIf(idxType, to);
    step = utils.castIf(idxType, step);
    
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
                            loc, utils.castIf(t, ph), direction
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

antlrcpp::Any DaphneDSLVisitor::visitArgExpr(DaphneDSLGrammarParser::ArgExprContext * ctx) {
    // Retrieve the name of the referenced CLI argument.
    std::string arg = ctx->arg->getText();
    
    // Find out if this argument was specified on the comman line.
    auto it = args.find(arg);
    if(it == args.end())
        throw std::runtime_error(
                "argument " + arg + " referenced, but not provided as a "
                "command line argument"
        );

    // Parse the string that was passed as the value for this argument on the
    // command line as a DaphneDSL literal.
    std::istringstream stream(it->second);
    antlr4::ANTLRInputStream input(stream);
    input.name = "argument"; // TODO Does this make sense?
    DaphneDSLGrammarLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    DaphneDSLGrammarParser parser(&tokens);
    DaphneDSLGrammarParser::LiteralContext * literalCtx = parser.literal();
    return visitLiteral(literalCtx);
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
    return utils.valueOrError(visit(ctx->expr()));
}

antlrcpp::Any DaphneDSLVisitor::visitCallExpr(DaphneDSLGrammarParser::CallExprContext * ctx) {
    std::string func = ctx->func->getText();
    mlir::Location loc = builder.getUnknownLoc();
 
    // Parse arguments.
    std::vector<mlir::Value> args;
    for(unsigned i = 0; i < ctx->expr().size(); i++)
        args.push_back(utils.valueOrError(visit(ctx->expr(i))));
    
    // Create DaphneIR operation for the built-in function.
    return builtins.build(loc, func, args);
}

antlrcpp::Any DaphneDSLVisitor::visitCastExpr(DaphneDSLGrammarParser::CastExprContext * ctx) {
    mlir::Type resType;
    
    if(ctx->DATA_TYPE()) {
        std::string dtStr = ctx->DATA_TYPE()->getText();
        if(dtStr == "matrix") {
            mlir::Type vt;
            if(ctx->VALUE_TYPE())
                vt = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
            else
                vt = utils.unknownType;
            resType = utils.matrixOf(vt);
        }
        else if(dtStr == "frame")
            throw std::runtime_error("casting to a frame is not supported yet");
        else
            throw std::runtime_error(
                    "unsupported data type in cast expression: " + dtStr
            );
    }
    else if(ctx->VALUE_TYPE())
        resType = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
    else
        throw std::runtime_error(
                "casting requires the specification of the target data and/or "
                "value type"
        );
    
    return static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(
            builder.getUnknownLoc(),
            resType,
            utils.valueOrError(visit(ctx->expr()))
    ));
}

// TODO Reduce the code duplication with visitRightIdxExtractExpr.
antlrcpp::Any DaphneDSLVisitor::visitRightIdxFilterExpr(DaphneDSLGrammarParser::RightIdxFilterExprContext * ctx) {
    mlir::Value obj = utils.valueOrError(visit(ctx->obj));
    mlir::Type objType = obj.getType();
    if(ctx->rows && ctx->cols)
        throw std::runtime_error(
                "currently right indexing supports either rows or columns, "
                "but not both at the same time"
        );
    if(ctx->rows) {
        mlir::Value rows = utils.valueOrError(visit(ctx->rows));
        return static_cast<mlir::Value>(builder.create<mlir::daphne::FilterRowOp>(
                builder.getUnknownLoc(), objType, obj, rows
        ));
    }
    if(ctx->cols)
        throw std::runtime_error(
                "currently right indexing (for filter) supports only rows"
        );
    throw std::runtime_error(
            "right indexing requires the specification of rows and/or columns"
    );
}

// TODO Reduce the code duplication with visitRightIdxFilterExpr.
antlrcpp::Any DaphneDSLVisitor::visitRightIdxExtractExpr(DaphneDSLGrammarParser::RightIdxExtractExprContext * ctx) {
    mlir::Value obj = utils.valueOrError(visit(ctx->obj));
    mlir::Type objType = obj.getType();
    if(ctx->rows && ctx->cols)
        throw std::runtime_error(
                "currently right indexing supports either rows or columns, "
                "but not both at the same time"
        );
    if(ctx->rows) {
        mlir::Value rows = utils.valueOrError(visit(ctx->rows));
        return static_cast<mlir::Value>(builder.create<mlir::daphne::ExtractRowOp>(
                builder.getUnknownLoc(), objType, obj, rows
        ));
    }
    if(ctx->cols) {
        mlir::Value cols = utils.valueOrError(visit(ctx->cols));
        mlir::Type colsType = cols.getType();
        // TODO Consider all supported value types.
        if(colsType.isInteger(64) || colsType.isF64()) {
            cols = utils.castSizeIf(cols);
            colsType = cols.getType();
        }
        mlir::Type resType;
        if(objType.isa<mlir::daphne::MatrixType>())
            // Data type and value type remain the same.
            resType = objType;
        else if(objType.isa<mlir::daphne::FrameType>())
            // Data type remains the same, but the value type of the result's
            // single column is currently unknown.
            // TODO If the column is selected by position, we could know its
            // type already here.
            resType = mlir::daphne::FrameType::get(
                    builder.getContext(), {utils.unknownType}
            );
        return static_cast<mlir::Value>(builder.create<mlir::daphne::ExtractColOp>(
                builder.getUnknownLoc(), resType, obj, cols
        ));
    }
    // TODO Actually, this would be okay, but we should think about whether
    // it should be a no-op or a copy.
    throw std::runtime_error(
            "right indexing requires the specification of rows and/or columns"
    );
}

antlrcpp::Any DaphneDSLVisitor::visitMatmulExpr(DaphneDSLGrammarParser::MatmulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "@")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MatMulOp>(loc, lhs.getType(), lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitPowExpr(DaphneDSLGrammarParser::PowExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "^")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwPowOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitModExpr(DaphneDSLGrammarParser::ModExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "%")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwModOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMulExpr(DaphneDSLGrammarParser::MulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "*")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwMulOp>(loc, lhs, rhs));
    if(op == "/")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwDivOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitAddExpr(DaphneDSLGrammarParser::AddExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "+")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAddOp>(loc, lhs, rhs));
    if(op == "-")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwSubOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCmpExpr(DaphneDSLGrammarParser::CmpExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
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

antlrcpp::Any DaphneDSLVisitor::visitConjExpr(DaphneDSLGrammarParser::ConjExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "&&")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAndOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitDisjExpr(DaphneDSLGrammarParser::DisjExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "||")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwOrOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitLiteral(DaphneDSLGrammarParser::LiteralContext * ctx) {
    // TODO The creation of the ConstantOps could be simplified: We don't need
    // to create attributes here, since there are custom builder methods for
    // primitive C++ data types.
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
        const std::string litStr = lit->getText();
        double val;
        if(litStr == "nan")
            val = std::numeric_limits<double>::quiet_NaN();
        else if(litStr == "inf")
            val = std::numeric_limits<double>::infinity();
        else if(litStr == "-inf")
            val = -std::numeric_limits<double>::infinity();
        else
            val = atof(litStr.c_str());
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                        loc,
                        builder.getF64FloatAttr(val)
                )
        );
    }
    if(ctx->bl)
        return visit(ctx->bl);
    if(auto lit = ctx->STRING_LITERAL()) {
        std::string val = lit->getText();
        
        // Remove quotation marks.
        val = val.substr(1, val.size() - 2);
        
        // Replace escape sequences.
        val = std::regex_replace(val, std::regex(R"(\\b)"), "\b");
        val = std::regex_replace(val, std::regex(R"(\\f)"), "\f");
        val = std::regex_replace(val, std::regex(R"(\\n)"), "\n");
        val = std::regex_replace(val, std::regex(R"(\\r)"), "\r");
        val = std::regex_replace(val, std::regex(R"(\\t)"), "\t");
        val = std::regex_replace(val, std::regex(R"(\\\")"), "\"");
        val = std::regex_replace(val, std::regex(R"(\\\\)"), "\\");
        
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(loc, val)
        );
    }
    throw std::runtime_error("unexpected literal");
}

antlrcpp::Any DaphneDSLVisitor::visitBoolLiteral(DaphneDSLGrammarParser::BoolLiteralContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    bool val;
    if(ctx->KW_TRUE())
        val = true;
    else if(ctx->KW_FALSE())
        val = false;
    else
        throw std::runtime_error("unexpected bool literal");

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(
                loc,
                builder.getBoolAttr(val)
        )
    );
}