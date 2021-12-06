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
    
    mlir::Location loc = utils.getLoc(ctx->start);

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
    mlir::Location loc = utils.getLoc(ctx->start);
    
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
    mlir::Location loc = utils.getLoc(ctx->start);
    
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
    // TODO: fix for string literals when " are not escaped or not present
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
    mlir::Location loc = utils.getLoc(ctx->start);
 
    // Parse arguments.
    std::vector<mlir::Value> args;
    for(unsigned i = 0; i < ctx->expr().size(); i++)
        args.push_back(utils.valueOrError(visit(ctx->expr(i))));

    // search user defined functions
    auto range = functionsSymbolMap.equal_range(func);
    // TODO: find not only a matching version, but the `most` specialized
    for (auto it = range.first; it != range.second; ++it) {
        auto userDefinedFunc = it->second;
        auto funcTy = userDefinedFunc.getType();
        auto compatible = true;

        if (funcTy.getInputs().size() != args.size()) {
            continue;
        }
        for (auto compIt : llvm::zip(funcTy.getInputs(), args)) {
            auto funcInputType = std::get<0>(compIt);
            auto argVal = std::get<1>(compIt);

            auto funcMatTy = funcInputType.dyn_cast<mlir::daphne::MatrixType>();
            auto specializedMatTy = argVal.getType().dyn_cast<mlir::daphne::MatrixType>();
            bool isMatchingUnknownMatrix =
                funcMatTy && specializedMatTy && funcMatTy.getElementType() == utils.unknownType;
            if(funcInputType != argVal.getType() && !isMatchingUnknownMatrix && funcInputType != utils.unknownType) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            // TODO: variable results
            return builder
                .create<mlir::daphne::GenericCallOp>(loc,
                    userDefinedFunc.sym_name(),
                    args,
                    userDefinedFunc.getType().getResults())
                .getResult(0);
        }
    }
    if (range.second != range.first) {
        // FIXME: disallow user-defined function with same name as builtins, otherwise this would be wrong behaviour
        throw std::runtime_error("No function definition of `" + func + "` found with matching types");
    }

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
            utils.getLoc(ctx->start),
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
                utils.getLoc(ctx->rows->start), objType, obj, rows
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
                utils.getLoc(ctx->rows->start), objType, obj, rows
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
            utils.getLoc(ctx->cols->start), resType, obj, cols
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
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "@")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MatMulOp>(loc, lhs.getType(), lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitPowExpr(DaphneDSLGrammarParser::PowExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "^")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwPowOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitModExpr(DaphneDSLGrammarParser::ModExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "%")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwModOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMulExpr(DaphneDSLGrammarParser::MulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
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
    mlir::Location loc = utils.getLoc(ctx->op);
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
    mlir::Location loc = utils.getLoc(ctx->op);
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
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));
    
    if(op == "&&")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAndOp>(loc, lhs, rhs));
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitDisjExpr(DaphneDSLGrammarParser::DisjExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
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
    mlir::Location loc = utils.getLoc(ctx->start);
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
    mlir::Location loc = utils.getLoc(ctx->start);
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

void removeOperationsBeforeReturnOp(mlir::daphne::ReturnOp firstReturnOp, mlir::Block *block) {
    auto op = &block->getOperations().back();
    // erase in reverse order to ensure no uses will be left
    while(op != firstReturnOp) {
        auto prev = op->getPrevNode();
        op->emitWarning() << "Operation is ignored, as the function will return at " << firstReturnOp.getLoc();
        op->erase();
        op = prev;
    }
}

/**
 * @brief Ensures that the `caseBlock` has correct behaviour by appending operations, as the other case has an early return.
 *
 * @param ifOpWithEarlyReturn The old `IfOp` with the early return
 * @param caseBlock The new block for the case without a `ReturnOp`
 */
void rectifyIfCaseWithoutReturnOp(mlir::scf::IfOp ifOpWithEarlyReturn, mlir::Block *caseBlock) {
    // ensure there is a `YieldOp` (for later removal of such)
    if(caseBlock->empty() || !llvm::isa<mlir::scf::YieldOp>(caseBlock->back())) {
        mlir::OpBuilder builder(ifOpWithEarlyReturn->getContext());
        builder.setInsertionPoint(caseBlock, caseBlock->end());
        builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc());
    }

    // As this if-case doesn't have an early return we need to move/clone operations that should happen
    // into this case.
    auto opsAfterIf = ifOpWithEarlyReturn->getNextNode();
    while(opsAfterIf) {
        auto next = opsAfterIf->getNextNode();
        if(auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(opsAfterIf)) {
            auto parentOp = llvm::dyn_cast<mlir::scf::IfOp>(yieldOp->getParentOp());
            if(!parentOp) {
                throw std::runtime_error("Early return not nested in `if`s not yet supported!");
            }
            next = parentOp->getNextNode();
        }
        if(opsAfterIf->getBlock() == ifOpWithEarlyReturn->getBlock()) {
            // can be moved inside if
            opsAfterIf->moveBefore(caseBlock, caseBlock->end());
        }
        else {
            // can't move them directly, need clone (operations will be needed later)
            auto clonedOp = opsAfterIf->clone();
            mlir::OpBuilder builder(clonedOp->getContext());
            builder.setInsertionPoint(caseBlock, caseBlock->end());
            builder.insert(clonedOp);
        }
        opsAfterIf = next;
    }

    // Remove `YieldOp`s and replace the result values of `IfOp`s used by operations that got moved in
    // the previous loop with the correct values.
    auto currIfOp = ifOpWithEarlyReturn;
    auto currOp = &caseBlock->front();
    while(auto nextOp = currOp->getNextNode()) {
        if(auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(currOp)) {
            // cast was checked in previous loop
            for(auto it : llvm::zip(currIfOp.getResults(), yieldOp.getOperands())) {
                auto ifResult = std::get<0>(it);
                auto yieldedVal = std::get<1>(it);
                ifResult.replaceUsesWithIf(yieldedVal, [&](mlir::OpOperand &opOperand) {
                    return opOperand.getOwner()->getBlock() == caseBlock;
                });
            }
            currIfOp = llvm::dyn_cast_or_null<mlir::scf::IfOp>(currIfOp->getParentOp());
            yieldOp->erase();
        }
        currOp = nextOp;
    }
}

mlir::scf::YieldOp replaceReturnWithYield(mlir::daphne::ReturnOp returnOp) {
    mlir::OpBuilder builder(returnOp);
    auto yieldOp = builder.create<mlir::scf::YieldOp>(returnOp.getLoc(), returnOp.getOperands());
    returnOp->erase();
    return yieldOp;
}

void rectifyEarlyReturn(mlir::scf::IfOp ifOp, mlir::TypeRange resultTypes) {
    // FIXME: handle case where early return is in else block
    auto insertThenBlock = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        auto newThenBlock = nested.getBlock();
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), ifOp.thenBlock()->getOperations());

        auto returnOps = newThenBlock->getOps<mlir::daphne::ReturnOp>();
        if(!returnOps.empty()) {
            // NOTE: we ignore operations after return, could also throw an error
            removeOperationsBeforeReturnOp(*returnOps.begin(), newThenBlock);
        }
        else {
            rectifyIfCaseWithoutReturnOp(ifOp, newThenBlock);
        }
        auto returnOp = llvm::dyn_cast<mlir::daphne::ReturnOp>(newThenBlock->back());
        if(!returnOp) {
            // this should never happen, if it does check the `rectifyCaseByAppendingNecessaryOperations` function
            throw std::runtime_error("Final operation in then case has to be return op");
        }
        replaceReturnWithYield(returnOp);
    };
    auto insertElseBlock = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        auto newElseBlock = nested.getBlock();
        if(!ifOp.elseRegion().empty()) {
            newElseBlock->getOperations().splice(newElseBlock->end(), ifOp.elseBlock()->getOperations());
        }
        // TODO: check if already final operation is a return

        auto returnOps = newElseBlock->getOps<mlir::daphne::ReturnOp>();
        if(!returnOps.empty()) {
            // NOTE: we ignore operations after return, could also throw an error
            removeOperationsBeforeReturnOp(*returnOps.begin(), newElseBlock);
        }
        else {
            rectifyIfCaseWithoutReturnOp(ifOp, newElseBlock);
        }
        auto returnOp = llvm::dyn_cast<mlir::daphne::ReturnOp>(newElseBlock->back());
        if(!returnOp) {
            // this should never happen, if it does check the `rectifyCaseByAppendingNecessaryOperations` function
            throw std::runtime_error("Final operation in else case has to be return op");
        }
        replaceReturnWithYield(returnOp);
    };
    mlir::OpBuilder builder(ifOp);

    auto newIfOp = builder.create<mlir::scf::IfOp>(
        builder.getUnknownLoc(),
        resultTypes,
        ifOp.condition(),
        insertThenBlock,
        insertElseBlock
    );
    builder.create<mlir::daphne::ReturnOp>(ifOp->getLoc(), newIfOp.getResults());
    ifOp.erase();
}

/**
 * @brief Adapts the block such that only a single return at the end of the block is present, by moving early returns in
 * SCF-Ops.
 *
 * General procedure is finding the most nested early return and then SCF-Op by SCF-Op moves the return outside,
 * putting the case without early return into the other case. This is repeated until all SCF-Ops are valid and
 * only a final return exists. Might duplicate operations if we have more nested if ops like this example:
 * ```
 * if (a > 5) {
 *   if (a > 10) {
 *     return SOMETHING_A;
 *   }
 *   print("a > 5");
 * }
 * else {
 *   print("a <= 5");
 * }
 * print("no early return");
 * return SOMETHING_B;
 * ```
 * would be converted to (MLIR pseudo code)
 * ```
 * return scf.if(a > 5) {
 *   yield scf.if(a > 10) {
 *     yield SOMETHING_A;
 *   } else {
 *     print("a > 5");
 *     print("no early return"); // duplicated
 *     yield SOMETHING_B; // duplicated
 *   }
 * } else {
 *   print("a <= 5");
 *   print("no early return");
 *   yield SOMETHING_B;
 * }
 * ```
 *
 * @param funcBlock The block of the function with possible early returns
 */
void rectifyEarlyReturns(mlir::Block *funcBlock) {
    if(funcBlock->empty())
        return;
    while(true) {
        size_t levelOfMostNested = 0;
        mlir::daphne::ReturnOp mostNestedReturn;
        funcBlock->walk([&](mlir::daphne::ReturnOp returnOp) {
            size_t nested = 1;
            auto op = returnOp.getOperation();
            while(op->getBlock() != funcBlock) {
                ++nested;
                op = op->getParentOp();
            }

            if(nested > mostNestedReturn) {
                mostNestedReturn = returnOp;
                levelOfMostNested = nested;
            }
        });
        if(!mostNestedReturn || mostNestedReturn == &funcBlock->back()) {
            // finished!
            break;
        }

        auto parentOp = mostNestedReturn->getParentOp();
        if(auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(parentOp)) {
            rectifyEarlyReturn(ifOp, mostNestedReturn->getOperandTypes());
        }
        else {
            throw std::runtime_error(
                "Early return in `" + parentOp->getName().getStringRef().str() + "` is not supported.");
        }
    }
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionStatement(DaphneDSLGrammarParser::FunctionStatementContext *ctx) {
    auto loc = utils.getLoc(ctx->start);
    // TODO: check that the function does not shadow a builtin
    auto functionName = ctx->name->getText();
    // TODO: global variables support in functions
    auto globalSymbolTable = symbolTable;
    symbolTable = ScopedSymbolTable();

    // TODO: better check?
    if(globalSymbolTable.getNumScopes() > 1) {
        // TODO: create a function/class for throwing errors
        std::string s;
        llvm::raw_string_ostream stream(s);
        stream << loc << ": Functions can only be defined at top-level";
        throw std::runtime_error(s);
    }

    std::vector<std::string> funcArgNames;
    std::vector<mlir::Type> funcArgTypes;
    if(ctx->args) {
        auto functionArguments = static_cast<std::vector<std::pair<std::string, mlir::Type>>>(visit(ctx->args));
        for(const auto &pair : functionArguments) {
            if(std::find(funcArgNames.begin(), funcArgNames.end(), pair.first) != funcArgNames.end()) {
                throw std::runtime_error("Function argument name `" + pair.first + "` is used twice.");
            }
            funcArgNames.push_back(pair.first);
            funcArgTypes.push_back(pair.second);
        }
    }

    auto funcBlock = new mlir::Block();
    for(auto it : llvm::zip(funcArgNames, funcArgTypes)) {
        auto blockArg = funcBlock->addArgument(std::get<1>(it));
        handleAssignmentPart(std::get<0>(it), symbolTable, blockArg);
    }

    mlir::Type returnType;
    mlir::FuncOp functionOperation;
    if(ctx->retTy) {
        // early creation of FuncOp for recursion
        returnType = visit(ctx->retTy);
        functionOperation = createUserDefinedFuncOp(loc,
            builder.getFunctionType(funcArgTypes, {returnType}),
            functionName);
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(funcBlock);
    visitBlockStatement(ctx->bodyStmt);

    rectifyEarlyReturns(funcBlock);
    if(funcBlock->getOperations().empty()
        || !funcBlock->getOperations().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.create<mlir::daphne::ReturnOp>(utils.getLoc(ctx->stop));
    }

    auto returnOpTypes = funcBlock->getTerminator()->getOperandTypes();
    if(!functionOperation) {
        // late creation if no return types defined
        functionOperation = createUserDefinedFuncOp(loc,
            builder.getFunctionType(funcArgTypes, returnOpTypes),
            functionName);
    }
    else if(returnOpTypes != mlir::TypeRange({returnType})) {
        throw std::runtime_error(
            "Function `" + functionName + "` returns different type than specified in the definition");
    }
    functionOperation.body().push_front(funcBlock);

    symbolTable = globalSymbolTable;
    return functionOperation;
}

mlir::FuncOp DaphneDSLVisitor::createUserDefinedFuncOp(const mlir::Location &loc,
                                                       const mlir::FunctionType &funcType,
                                                       const std::string &functionName) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *moduleBody = module.getBody();
    auto functionSymbolName = utils.getUniqueFunctionSymbol(functionName);

    builder.setInsertionPoint(moduleBody, moduleBody->begin());
    auto functionOperation = builder.create<mlir::FuncOp>(loc, functionSymbolName, funcType);
    functionsSymbolMap.insert({functionName, functionOperation});
    return functionOperation;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArgs(DaphneDSLGrammarParser::FunctionArgsContext *ctx) {
    std::vector<std::pair<std::string, mlir::Type>> functionArguments;
    for(auto funcArgCtx: ctx->functionArg()) {
        functionArguments.push_back(visitFunctionArg(funcArgCtx));
    }
    return functionArguments;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArg(DaphneDSLGrammarParser::FunctionArgContext *ctx) {
    auto ty = utils.unknownType;
    if(ctx->ty) {
        ty = visitFuncTypeDef(ctx->ty);
    }
    return std::make_pair(ctx->var->getText(), ty);
}

antlrcpp::Any DaphneDSLVisitor::visitFuncTypeDef(DaphneDSLGrammarParser::FuncTypeDefContext *ctx) {
    auto type = utils.unknownType;
    if(ctx->dataTy) {
        std::string dtStr = ctx->dataTy->getText();
        if(dtStr == "matrix") {
            mlir::Type vt;
            if(ctx->elTy)
                vt = utils.getValueTypeByName(ctx->elTy->getText());
            else
                vt = utils.unknownType;
            type = utils.matrixOf(vt);
        }
        else {
            // TODO: should we do this?
            // auto loc = utils.getLoc(ctx->start);
            // emitError(loc) << "unsupported data type for function argument: " + dtStr;
            throw std::runtime_error(
                "unsupported data type for function argument: " + dtStr
            );
        }
    }
    else if(ctx->scalarTy)
        type = utils.getValueTypeByName(ctx->scalarTy->getText());
    return type;
}

antlrcpp::Any DaphneDSLVisitor::visitReturnStatement(DaphneDSLGrammarParser::ReturnStatementContext *ctx) {
    std::vector<mlir::Value> returns;
    for(auto expr: ctx->expr()) {
        returns.push_back(utils.valueOrError(visit(expr)));
    }
    return builder.create<mlir::daphne::ReturnOp>(utils.getLoc(ctx->start), returns);
}
