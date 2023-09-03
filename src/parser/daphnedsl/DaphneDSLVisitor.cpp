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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <parser/daphnedsl/DaphneDSLVisitor.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include <parser/CancelingErrorListener.h>
#include <parser/ScopedSymbolTable.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include "antlr4-runtime.h"
#include "DaphneDSLGrammarLexer.h"
#include "DaphneDSLGrammarParser.h"

#include <mlir/Dialect/SCF/IR/SCF.h>

#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <filesystem>

// ****************************************************************************
// Utilities
// ****************************************************************************

void DaphneDSLVisitor::handleAssignmentPart(
        const std::string & var,
        DaphneDSLGrammarParser::IndexingContext * idxCtx,
        ScopedSymbolTable & symbolTable,
        mlir::Value val
) {
    if(symbolTable.has(var) && symbolTable.get(var).isReadOnly)
        throw std::runtime_error("trying to assign read-only variable " + var);

    if(idxCtx) { // left indexing `var[idxCtx] = val;`
        if(!symbolTable.has(var))
            throw std::runtime_error(
                    "cannot use left indexing on variable " + var +
                    " before a value has been assigned to it"
            );
        mlir::Value obj = symbolTable.get(var).value;

        auto indexing = visit(idxCtx).as<std::pair<
                std::pair<bool, antlrcpp::Any>,
                std::pair<bool, antlrcpp::Any>
        >>();
        auto rows = indexing.first;
        auto cols = indexing.second;

        // TODO Use location of rows/cols in utils.getLoc(...) for better
        // error messages.
        if(rows.first && cols.first) {
            // TODO Use a combined InsertOp (row+col) (see #238).
            mlir::Value rowSeg = applyRightIndexing<
                    mlir::daphne::ExtractRowOp, mlir::daphne::SliceRowOp, mlir::daphne::NumRowsOp
            >(utils.getLoc(idxCtx->start), obj, rows.second, false);
            rowSeg = applyLeftIndexing<
                    mlir::daphne::InsertColOp,
                    mlir::daphne::NumColsOp
            >(utils.getLoc(idxCtx->start), rowSeg, val, cols.second, obj.getType().isa<mlir::daphne::FrameType>());
            obj = applyLeftIndexing<
                    mlir::daphne::InsertRowOp,
                    mlir::daphne::NumRowsOp
            >(utils.getLoc(idxCtx->start), obj, rowSeg, rows.second, false);
        }
        else if(rows.first) // rows specified
            obj = applyLeftIndexing<
                    mlir::daphne::InsertRowOp,
                    mlir::daphne::NumRowsOp
            >(utils.getLoc(idxCtx->start), obj, val, rows.second, false);
        else if(cols.first) // cols specified
            obj = applyLeftIndexing<
                    mlir::daphne::InsertColOp,
                    mlir::daphne::NumColsOp
            >(utils.getLoc(idxCtx->start), obj, val, cols.second, obj.getType().isa<mlir::daphne::FrameType>());
        else
            obj = val;

        symbolTable.put(var, ScopedSymbolTable::SymbolInfo(obj, false));
    }
    else // no left indexing `var = val;`
        symbolTable.put(var, ScopedSymbolTable::SymbolInfo(val, false));
}

template<class ExtractAxOp, class SliceAxOp, class NumAxOp>
mlir::Value DaphneDSLVisitor::applyRightIndexing(mlir::Location loc, mlir::Value arg, antlrcpp::Any ax, bool allowLabel) {
    if(ax.is<mlir::Value>()) { // indexing with a single SSA value (no ':')
        mlir::Value axVal = ax.as<mlir::Value>();
        if(CompilerUtils::hasObjType(axVal)) // data object
            return utils.retValWithInferedType(
                    builder.create<ExtractAxOp>(loc, utils.unknownType, arg, axVal)
            );
        else if(axVal.getType().isa<mlir::daphne::StringType>()) { // string
            if(allowLabel)
                return utils.retValWithInferedType(
                        builder.create<ExtractAxOp>(loc, utils.unknownType, arg, axVal)
                );
            else
                throw std::runtime_error(
                        "cannot use right indexing with label in this case"
                );
        }
        else // scalar
            return utils.retValWithInferedType(
                    builder.create<SliceAxOp>(
                            loc, utils.unknownType, arg,
                            utils.castSizeIf(axVal),
                            utils.castSizeIf(
                                    builder.create<mlir::daphne::EwAddOp>(
                                            loc, builder.getIntegerType(64, false),
                                            utils.castSI64If(axVal),
                                            builder.create<mlir::daphne::ConstantOp>(
                                                    loc, static_cast<int64_t>(1)
                                            )
                                    )
                            )
                    )
            );
    }
    else if(ax.is<std::pair<mlir::Value, mlir::Value>>()) { // indexing with a range (':')
        auto axPair = ax.as<std::pair<mlir::Value, mlir::Value>>();
        auto axLowerIncl = axPair.first;
        auto axUpperExcl = axPair.second;

        // Use defaults if lower or upper bound not specified.
        if(axLowerIncl == nullptr)
            axLowerIncl = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(0));
        if(axUpperExcl == nullptr)
            axUpperExcl = builder.create<NumAxOp>(loc, utils.sizeType, arg);

        return utils.retValWithInferedType(
                builder.create<SliceAxOp>(
                        loc, utils.unknownType, arg,
                        utils.castSizeIf(axLowerIncl),
                        utils.castSizeIf(axUpperExcl)
                )
        );
    }
    else
        throw std::runtime_error("unsupported type for right indexing");
}

template<class InsertAxOp, class NumAxOp>
mlir::Value DaphneDSLVisitor::applyLeftIndexing(mlir::Location loc, mlir::Value arg, mlir::Value ins, antlrcpp::Any ax, bool allowLabel) {
    mlir::Type argType = arg.getType();

    if(ax.is<mlir::Value>()) { // indexing with a single SSA value (no ':')
        mlir::Value axVal = ax.as<mlir::Value>();
        if(CompilerUtils::hasObjType(axVal)) // data object
            throw std::runtime_error(
                    "left indexing with positions as a data object is not supported (yet)"
            );
        else if(axVal.getType().isa<mlir::daphne::StringType>()) { // string
            if(allowLabel)
                // TODO Support this (#239).
                throw std::runtime_error("left indexing by label is not supported yet");
//                return static_cast<mlir::Value>(
//                        builder.create<InsertAxOp>(loc, argType, arg, ins, axVal)
//                );
            else
                throw std::runtime_error(
                        "cannot use left indexing with label in this case"
                );
        }
        else // scalar
            return static_cast<mlir::Value>(
                    builder.create<InsertAxOp>(
                            loc, argType, arg, ins,
                            utils.castSizeIf(axVal),
                            utils.castSizeIf(
                                    builder.create<mlir::daphne::EwAddOp>(
                                            loc, builder.getIntegerType(64, false),
                                            utils.castSI64If(axVal),
                                            builder.create<mlir::daphne::ConstantOp>(
                                                    loc, static_cast<int64_t>(1)
                                            )
                                    )
                            )
                    )
            );
    }
    else if(ax.is<std::pair<mlir::Value, mlir::Value>>()) { // indexing with a range (':')
        auto axPair = ax.as<std::pair<mlir::Value, mlir::Value>>();
        auto axLowerIncl = axPair.first;
        auto axUpperExcl = axPair.second;

        // Use defaults if lower or upper bound not specified.
        if(axLowerIncl == nullptr)
            axLowerIncl = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(0));
        if(axUpperExcl == nullptr)
            axUpperExcl = builder.create<NumAxOp>(loc, utils.sizeType, arg);

        return static_cast<mlir::Value>(
                builder.create<InsertAxOp>(
                        loc, argType, arg, ins,
                        utils.castSizeIf(axLowerIncl),
                        utils.castSizeIf(axUpperExcl)
                )
        );
    }
    else
        throw std::runtime_error("unsupported type for left indexing");
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

antlrcpp::Any DaphneDSLVisitor::visitImportStatement(DaphneDSLGrammarParser::ImportStatementContext * ctx) {
    if(symbolTable.getNumScopes() != 1)
        throw std::runtime_error("Imports can only be done in the main scope");

    const char prefixDelim = '.';
    std::string prefix;
    std::vector<std::string> importPaths;
    std::string path = ctx->filePath->getText();
    // Remove quotes
    path = path.substr(1, path.size() - 2);
    
    std::filesystem::path importerDirPath = std::filesystem::absolute(std::filesystem::path(scriptPaths.top())).parent_path();
    std::filesystem::path importingPath = path;

    //Determine the prefix from alias/filename
    if(ctx->alias)
    {
        prefix = ctx->alias->getText();
        prefix = prefix.substr(1, prefix.size() - 2);
    }
    else
        prefix = importingPath.stem().string(); 

    prefix += prefixDelim;

    // Absolute path can be used as is, we have to handle relative paths and config paths
    if(importingPath.is_relative())
    {
        std::filesystem::path absolutePath = importerDirPath /  importingPath;
        if(std::filesystem::exists(absolutePath))
            absolutePath = std::filesystem::canonical(absolutePath);

        // Check directories in UserConfig (if provided)
        if(!userConf.daphnedsl_import_paths.empty())
        {
            const auto& configPaths = userConf.daphnedsl_import_paths;
            // User specified _default_ paths.
            if(importingPath.has_extension() && (configPaths.find("default_dirs") != configPaths.end()))
            {
                for(std::filesystem::path defaultPath: configPaths.at("default_dirs"))
                {
                    std::filesystem::path libFile = defaultPath / path;
                    if(std::filesystem::exists(libFile))
                    {
                        if(std::filesystem::exists(absolutePath) && std::filesystem::canonical(libFile) != absolutePath)
                            throw std::runtime_error(std::string("Ambiguous import: ").append(importingPath)
                                .append(", found another file with the same name in default paths of UserConfig: ").append(libFile));
                        absolutePath = libFile;
                    }
                }
            }

            // User specified "libraries" -> import all files
            if(!importingPath.has_extension() && (configPaths.find(path) != configPaths.end()))
                for (std::filesystem::path const& dir_entry : std::filesystem::directory_iterator{configPaths.at(path)[0]})
                    importPaths.push_back(dir_entry.string());
        }
        path = absolutePath.string();
    }

    if(importPaths.empty())
        importPaths.push_back(path);

    if(std::filesystem::absolute(scriptPaths.top()).string() == path)
        throw std::runtime_error(std::string("You cannot import the file you are currently in: ").append(path));

    for(const auto& somePath: importPaths)
    {
        for(const auto& imported: importedFiles)
            if(std::filesystem::equivalent(somePath, imported))
                throw std::runtime_error(std::string("You cannot import the same file twice: ").append(somePath));

        importedFiles.push_back(somePath);
    }

    antlrcpp::Any res;
    for(const auto& importPath : importPaths)
    {
        if(!std::filesystem::exists(importPath))
            throw std::runtime_error(std::string("The import path doesn't exist: ").append(importPath));

        std::string finalPrefix = prefix;
        auto origScope = symbolTable.extractScope();

        // If we import a library, we insert a filename (e.g., "algorithms/kmeans.daphne" -> algorithms.kmeans.km)
        if(!importingPath.has_extension())
            finalPrefix += std::filesystem::path(importPath).stem().string() + prefixDelim;
        else 
        {
            // If the prefix is already occupied (and is not part of some other prefix), we append a parent directory name
            for(const auto& symbol : origScope)
                if(symbol.first.find(finalPrefix) == 0 && std::count(symbol.first.begin(), symbol.first.end(), '.') == 1)
                {
                    // Throw error when we want to use an explicit alias that results in a prefix clash
                    if(ctx->alias)
                    {
                        throw std::runtime_error(std::string("Alias ").append(ctx->alias->getText())
                        .append(" results in a name clash with another prefix"));
                    }
//                    finalPrefix = importingPath.parent_path().getFilename().string() + prefixDelim + finalPrefix;
                    finalPrefix.insert(0, importingPath.parent_path().filename().string() + prefixDelim);
                    break;
                }
        }
        
        CancelingErrorListener errorListener;
        std::ifstream ifs(importPath, std::ios::in);
        antlr4::ANTLRInputStream input(ifs);
        input.name = importPath;

        DaphneDSLGrammarLexer lexer(&input);
        lexer.removeErrorListeners();
        lexer.addErrorListener(&errorListener);
        antlr4::CommonTokenStream tokens(&lexer);

        DaphneDSLGrammarParser parser(&tokens);
        parser.removeErrorListeners();
        parser.addErrorListener(&errorListener);
        DaphneDSLGrammarParser::ScriptContext * importCtx = parser.script();
        
        std::multimap<std::string, mlir::func::FuncOp> origFuncMap = functionsSymbolMap;
        functionsSymbolMap.clear();

        symbolTable.pushScope();
        scriptPaths.push(path);
        res = visitScript(importCtx);
        scriptPaths.pop();

        ScopedSymbolTable::SymbolTable symbTable = symbolTable.extractScope();

        // If the current import file also imported something, we discard it
        for(const auto& symbol : symbTable)
            if(symbol.first.find('.') == std::string::npos)
                origScope[finalPrefix + symbol.first] = symbol.second;

        symbolTable.put(origScope);
        
        for(std::pair<std::string, mlir::func::FuncOp> funcSymbol : functionsSymbolMap)
            if(funcSymbol.first.find('.') == std::string::npos)
                origFuncMap.insert({finalPrefix + funcSymbol.first, funcSymbol.second});
        functionsSymbolMap.clear();
        functionsSymbolMap = origFuncMap;
    }
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
                ctx->indexing(0),
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
                            ctx->IDENTIFIER(i)->getText(), ctx->indexing(i), symbolTable, rhsAsRR[i]
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
    return nullptr;
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

    // Generate the operations for the else-block, if it is present. Otherwise,
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
    std::vector<mlir::Value> resultsThen;
    std::vector<mlir::Value> resultsElse;
    for(auto it = owUnion.begin(); it != owUnion.end(); it++) {
        mlir::Value valThen = symbolTable.get(*it, owThen).value;
        mlir::Value valElse = symbolTable.get(*it, owElse).value;
        if(valThen.getType() != valElse.getType()) {
            // TODO We could try to cast the types.
            std::string s;
            llvm::raw_string_ostream stream(s);
            loc.print(stream);
            stream << ":\n    ";
            valThen.print(stream);
            stream << "\n      is not equal to \n    ";
            valElse.print(stream);
            throw std::runtime_error(fmt::format("in {}:{}:\n  type mismatch near script location: {}",
                    __FILE__, __LINE__, stream.str()));
        }
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
            cond,
            insertThenBlockDo,
            (ctx->elseStmt || !owUnion.empty()) ? insertElseBlockDo : insertElseBlockNo
    );

    // Rewire the results of the if-operation to their variable names.
    size_t i = 0;
    for(auto it = owUnion.begin(); it != owUnion.end(); it++)
        symbolTable.put(*it, ScopedSymbolTable::SymbolInfo(ifOp.getResults()[i++], false));

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

        beforeBlock->addArgument(type, builder.getUnknownLoc());
        afterBlock->addArgument(type, builder.getUnknownLoc());
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
    whileOp.getBefore().push_back(beforeBlock);
    whileOp.getAfter().push_back(afterBlock);

    size_t i = 0;
    for(auto it = ow.begin(); it != ow.end(); it++) {
        // Replace usages of the variables updated in the loop's body by the
        // corresponding block arguments.
        whileOperands[i].replaceUsesWithIf(beforeBlock->getArgument(i), [&](mlir::OpOperand & operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.getBefore().isAncestor(parentRegion);
        });
        whileOperands[i].replaceUsesWithIf(afterBlock->getArgument(i), [&](mlir::OpOperand & operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.getAfter().isAncestor(parentRegion);
        });

        // Rewire the results of the WhileOp to their variable names.
        symbolTable.put(it->first, ScopedSymbolTable::SymbolInfo(whileOp.getResults()[i++], false));
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
                builder.create<mlir::daphne::ConstantOp>(loc, t, builder.getIntegerAttr(t, -1)),
                builder.create<mlir::daphne::EwMulOp>(
                        loc,
                        builder.create<mlir::daphne::ConstantOp>(loc, t, builder.getIntegerAttr(t, 2)),
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
    mlir::Value ph = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(123));
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
        symbolTable.put(it->first, ScopedSymbolTable::SymbolInfo(forOp.getResults()[i], false));

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

    // Find out if this argument was specified on the command line.
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
    std::string var;
    const auto& identifierVec = ctx->IDENTIFIER();
    for(size_t s = 0; s < identifierVec.size(); s++)
        var += (s < identifierVec.size() - 1) ? identifierVec[s]->getText() + '.' : identifierVec[s]->getText();

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

bool DaphneDSLVisitor::argAndUDFParamCompatible(mlir::Type argTy, mlir::Type paramTy) const {
    auto argMatTy = argTy.dyn_cast<mlir::daphne::MatrixType>();
    auto paramMatTy = paramTy.dyn_cast<mlir::daphne::MatrixType>();

    // TODO This is rather a workaround than a thorough solution, since
    // unknown argument types do not really allow to check compatibility.
    
    // Argument type and parameter type are compatible if...
    return
        // ...they are the same, OR
        paramTy == argTy ||
        // ...at least one of them is unknown, OR
        argTy == utils.unknownType || paramTy == utils.unknownType ||
        // ...they are both matrices and at least one of them is of unknown value type.
        (argMatTy && paramMatTy && (
            argMatTy.getElementType() == utils.unknownType ||
            paramMatTy.getElementType() == utils.unknownType
        ));
}

std::optional<mlir::func::FuncOp> DaphneDSLVisitor::findMatchingUDF(const std::string &functionName, const std::vector<mlir::Value> &args) const {
    // search user defined functions
    auto range = functionsSymbolMap.equal_range(functionName);
    // TODO: find not only a matching version, but the `most` specialized
    for (auto it = range.first; it != range.second; ++it) {
        auto userDefinedFunc = it->second;
        auto funcTy = userDefinedFunc.getFunctionType();
        auto compatible = true;

        if (funcTy.getNumInputs() != args.size()) {
            continue;
        }
        for (auto compIt : llvm::zip(funcTy.getInputs(), args)) {
            auto funcParamType = std::get<0>(compIt);
            auto argVal = std::get<1>(compIt);

            if (!argAndUDFParamCompatible(argVal.getType(), funcParamType)) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            return userDefinedFunc;
        }
    }
    // UDF with the provided name exists, but no version matches the argument types
    if (range.second != range.first) {
        // FIXME: disallow user-defined function with same name as builtins, otherwise this would be wrong behaviour
        throw std::runtime_error("No function definition of `" + functionName + "` found with matching types");
    }

    // UDF with the provided name does not exist
    return std::nullopt;
}


std::optional<mlir::func::FuncOp> DaphneDSLVisitor::findMatchingUnaryUDF(const std::string &functionName, mlir::Type argType) const {
    // search user defined functions
    auto range = functionsSymbolMap.equal_range(functionName);
    
    // TODO: find not only a matching version, but the `most` specialized    
    for (auto it = range.first; it != range.second; ++it) {
        auto userDefinedFunc = it->second;
        auto funcTy = userDefinedFunc.getFunctionType();

        if (funcTy.getNumInputs() != 1) {
            continue;
        }

        if (argAndUDFParamCompatible(argType, funcTy.getInput(0))) {
            return userDefinedFunc;
        }
    }
    // UDF with the provided name exists, but no version matches the argument types
    if (range.second != range.first) {
        // FIXME: disallow user-defined function with same name as builtins, otherwise this would be wrong behaviour
        throw std::runtime_error("No function definition of `" + functionName + "` found with matching types");
    }

    // UDF with the provided name does not exist
    return std::nullopt;
}

antlrcpp::Any DaphneDSLVisitor::handleMapOpCall(DaphneDSLGrammarParser::CallExprContext * ctx) {
    std::string func;
    const auto& identifierVec = ctx->IDENTIFIER();
    for(size_t s = 0; s < identifierVec.size(); s++)
        func += (s < identifierVec.size() - 1) ? identifierVec[s]->getText() + '.' : identifierVec[s]->getText();
    assert(func == "map" && "Called 'handleMapOpCall' for another function");
    mlir::Location loc = utils.getLoc(ctx->start);
    
    if (ctx->expr().size() != 2) {
        throw std::runtime_error(
                "built-in function 'map' expects exactly 2 argument(s), but got " +
                std::to_string(ctx->expr().size())
        );
    }

    std::vector<mlir::Value> args;
    
    auto argVal = utils.valueOrError(visit(ctx->expr(0)));
    args.push_back(argVal);

    auto argMatTy = argVal.getType().dyn_cast<mlir::daphne::MatrixType>();
    if (!argMatTy)
        throw std::runtime_error("built-in function 'map' expects argument of type matrix as its first parameter");
    
    std::string udfName = ctx->expr(1)->getText();
    auto maybeUDF = findMatchingUnaryUDF(udfName, argMatTy.getElementType());

    if (!maybeUDF)
        throw std::runtime_error("No function definition of `" + udfName + "` found");

    args.push_back(static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(loc, maybeUDF->getSymName().str())
    ));

    // Create DaphneIR operation for the built-in function.
    return builtins.build(loc, func, args);
}


antlrcpp::Any DaphneDSLVisitor::visitCallExpr(DaphneDSLGrammarParser::CallExprContext * ctx) {
    std::string func;
    const auto& identifierVec = ctx->IDENTIFIER();
    for(size_t s = 0; s < identifierVec.size(); s++)
        func += (s < identifierVec.size() - 1) ? identifierVec[s]->getText() + '.' : identifierVec[s]->getText();
    mlir::Location loc = utils.getLoc(ctx->start);
 
    if (func == "map") 
        return handleMapOpCall(ctx);

    // Parse arguments.
    std::vector<mlir::Value> args_vec;
    for(unsigned i = 0; i < ctx->expr().size(); i++)
        args_vec.push_back(utils.valueOrError(visit(ctx->expr(i))));

    auto maybeUDF = findMatchingUDF(func, args_vec);

    if (maybeUDF) {
        auto funcTy = maybeUDF->getFunctionType();
        auto co = builder
            .create<mlir::daphne::GenericCallOp>(loc,
                maybeUDF->getSymName(),
                args_vec,
                funcTy.getResults());
        if(funcTy.getNumResults() > 1)
            return co.getResults();
        else
            // If the UDF has no return values, the value returned here
            // is invalid. But that seems to be okay, since it is never
            // used as a mlir::Value in that case.
            return co.getResult(0);
    }

    // Create DaphneIR operation for the built-in function.
    return builtins.build(loc, func, args_vec);
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
            {
                vt = utils.valueOrError(visit(ctx->expr())).getType();
                if(vt.isa<mlir::daphne::FrameType>())
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    vt = vt.dyn_cast<mlir::daphne::FrameType>().getColumnTypes()[0];
                if(vt.isa<mlir::daphne::MatrixType>())
                    vt = vt.dyn_cast<mlir::daphne::MatrixType>().getElementType();
            }
            resType = utils.matrixOf(vt);
        }
        else if(dtStr == "frame") {
            // Currently does not support casts of type "Specify value type only" (e.g., as.si64(x)) 
            // and "Specify data type and value type" (e.g., as.frame<[si64, f64]>(x)) 
            std::vector<mlir::Type> colTypes;
            // TODO Take the number of columns into account.
            if(ctx->VALUE_TYPE())
                throw std::runtime_error("casting to a frame with particular column types is not supported yet");
                //colTypes = {utils.getValueTypeByName(ctx->VALUE_TYPE()->getText())};
            else {
                // TODO This fragment should be factored out, such that we can
                // reuse it for matrix/frame/scalar.
                mlir::Type argType = utils.valueOrError(visit(ctx->expr())).getType();
                if(argType.isa<mlir::daphne::MatrixType>())
                    colTypes = {argType.dyn_cast<mlir::daphne::MatrixType>().getElementType()};
                else if(argType.isa<mlir::daphne::FrameType>())
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    colTypes = {argType.dyn_cast<mlir::daphne::FrameType>().getColumnTypes()[0]};
                else
                    colTypes = {argType};
            }
            resType = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        }
        else if(dtStr == "scalar") {
            if(ctx->VALUE_TYPE())
                resType = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
            else
            {
                // TODO This fragment should be factored out, such that we can
                // reuse it for matrix/frame/scalar.
                mlir::Type argType = utils.valueOrError(visit(ctx->expr())).getType();
                if(argType.isa<mlir::daphne::MatrixType>())
                    resType = argType.dyn_cast<mlir::daphne::MatrixType>().getElementType();
                else if(argType.isa<mlir::daphne::FrameType>())
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    resType = argType.dyn_cast<mlir::daphne::FrameType>().getColumnTypes()[0];
                else
                    resType = argType;
            }
        }
        else
            throw std::runtime_error(
                    "unsupported data type in cast expression: " + dtStr
            );
    }
    else if(ctx->VALUE_TYPE())
    { // Data type shall be retained
        mlir::Type vt = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
        mlir::Type argTy = utils.valueOrError(visit(ctx->expr())).getType();
        if(argTy.isa<mlir::daphne::MatrixType>())
            resType = utils.matrixOf(vt);
        else if(argTy.isa<mlir::daphne::FrameType>())
        {
            throw std::runtime_error("casting to a frame with particular column types is not supported yet");
            //size_t numCols = argTy.dyn_cast<mlir::daphne::FrameType>().getColumnTypes().size();
            //std::vector<mlir::Type> colTypes(numCols, vt);
            //resType = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        }
        else
            resType = vt;
    }
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

antlrcpp::Any DaphneDSLVisitor::visitRightIdxFilterExpr(DaphneDSLGrammarParser::RightIdxFilterExprContext * ctx) {
    mlir::Value obj = utils.valueOrError(visit(ctx->obj));

    if(ctx->rows) // rows specified
        obj = builder.create<mlir::daphne::FilterRowOp>(
                utils.getLoc(ctx->rows->start),
                obj.getType(),
                obj,
                utils.valueOrError(visit(ctx->rows))
        );
    if(ctx->cols) // cols specified
        obj = builder.create<mlir::daphne::FilterColOp>(
                utils.getLoc(ctx->cols->start),
                obj.getType(), // TODO Not correct for frames, see #484.
                obj,
                utils.valueOrError(visit(ctx->cols))
        );

    // Note: If rows and cols are specified, we create two filter steps.
    // This can be inefficient, but it is simpler for now.
    // TODO Create a combined FilterOp

    // Note: If neither rows nor cols are specified, we simply return the
    // object.

    return obj;
}

antlrcpp::Any DaphneDSLVisitor::visitRightIdxExtractExpr(DaphneDSLGrammarParser::RightIdxExtractExprContext * ctx) {
    mlir::Value obj = utils.valueOrError(visit(ctx->obj));

    auto indexing = visit(ctx->idx).as<std::pair<
            std::pair<bool, antlrcpp::Any>,
            std::pair<bool, antlrcpp::Any>
    >>();
    auto rows = indexing.first;
    auto cols = indexing.second;

    // TODO Use location of rows/cols in utils.getLoc(...) for better
    // error messages.
    if(rows.first) // rows specified
        obj = applyRightIndexing<
                mlir::daphne::ExtractRowOp,
                mlir::daphne::SliceRowOp,
                mlir::daphne::NumRowsOp
        >(utils.getLoc(ctx->idx->start), obj, rows.second, false);
    if(cols.first) // cols specified
        obj = applyRightIndexing<
                mlir::daphne::ExtractColOp,
                mlir::daphne::SliceColOp,
                mlir::daphne::NumColsOp
        >(utils.getLoc(ctx->idx->start), obj, cols.second, obj.getType().isa<mlir::daphne::FrameType>());

    // Note: If rows and cols are specified, we create two extraction steps.
    // This can be inefficient, but it is simpler for now.
    // TODO Create a combined ExtractOp/SliceOp.

    // Note: If neither rows nor cols are specified, we simply return the
    // object.

    return obj;
}

antlrcpp::Any DaphneDSLVisitor::visitMatmulExpr(DaphneDSLGrammarParser::MatmulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "@") {
        mlir::Value f = builder.create<mlir::daphne::ConstantOp>(loc, false);
        return utils.retValWithInferedType(builder.create<mlir::daphne::MatMulOp>(
                loc, lhs.getType(), lhs, rhs, f, f
        ));
    }
    
    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitPowExpr(DaphneDSLGrammarParser::PowExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "^")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwPowOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitModExpr(DaphneDSLGrammarParser::ModExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "%")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwModOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMulExpr(DaphneDSLGrammarParser::MulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "*")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwMulOp>(loc, lhs, rhs));
    if(op == "/")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwDivOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitAddExpr(DaphneDSLGrammarParser::AddExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "+")
        // Note that we use '+' for both addition (EwAddOp) and concatenation
        // (ConcatOp). The choice is made based on the types of the operands
        // (if one operand is a string, we choose ConcatOp). However, the types
        // might not be known at this point in time. Thus, we always create an
        // EwAddOp here. Note that EwAddOp has a canonicalize method rewriting
        // it to ConcatOp if necessary.
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwAddOp>(loc, lhs, rhs));
    if(op == "-")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwSubOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCmpExpr(DaphneDSLGrammarParser::CmpExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "==")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwEqOp>(loc, lhs, rhs));
    if(op == "!=")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwNeqOp>(loc, lhs, rhs));
    if(op == "<")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwLtOp>(loc, lhs, rhs));
    if(op == "<=")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwLeOp>(loc, lhs, rhs));
    if(op == ">")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwGtOp>(loc, lhs, rhs));
    if(op == ">=")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwGeOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitConjExpr(DaphneDSLGrammarParser::ConjExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "&&")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwAndOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitDisjExpr(DaphneDSLGrammarParser::DisjExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(op == "||")
        return utils.retValWithInferedType(builder.create<mlir::daphne::EwOrOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCondExpr(DaphneDSLGrammarParser::CondExprContext * ctx) {
    return static_cast<mlir::Value>(builder.create<mlir::daphne::CondOp>(
            utils.getLoc(ctx->start),
            utils.unknownType,
            utils.valueOrError(visit(ctx->cond)),
            utils.valueOrError(visit(ctx->thenExpr)),
            utils.valueOrError(visit(ctx->elseExpr))
    ));
}

antlrcpp::Any DaphneDSLVisitor::visitMatrixLiteralExpr(DaphneDSLGrammarParser::MatrixLiteralExprContext * ctx) {
    if(!ctx->literal().size())
        throw std::runtime_error("can't infer type from an empty matrix");

    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value currentValue = utils.valueOrError(visitLiteral(ctx->literal(0)));
    mlir::Value result;
    mlir::Type valueType = currentValue.getType();

    // TODO Reduce the code duplication in these cases.
    if(currentValue.getType().isSignedInteger(64)){
        std::shared_ptr<int64_t[]> vals = std::shared_ptr<int64_t[]>(new int64_t[ctx->literal().size()]);
        for(unsigned i = 0; i < ctx->literal().size(); i++)
        {
            currentValue = utils.valueOrError(visitLiteral(ctx->literal(i)));
            if(currentValue.getType() != valueType)
                throw std::runtime_error("matrix of elements of different types");
            vals.get()[i] = CompilerUtils::constantOrThrow<int64_t>(
                    currentValue, "elements of matrix literals must be parse-time constants"
            );
        }
        auto mat = DataObjectFactory::create<DenseMatrix<int64_t>>(ctx->literal().size(), 1, vals);
        result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::MatrixConstantOp>(loc, utils.matrixOf(valueType),
                        builder.create<mlir::daphne::ConstantOp>(loc, reinterpret_cast<uint64_t>(mat))
                )
        );
    }
    else if(currentValue.getType().isF64()){
        std::shared_ptr<double[]> vals = std::shared_ptr<double[]>(new double[ctx->literal().size()]);
        for(unsigned i = 0; i < ctx->literal().size(); i++)
        {
            currentValue = utils.valueOrError(visitLiteral(ctx->literal(i)));
            if(currentValue.getType() != valueType)
                throw std::runtime_error("matrix of elements of different types");
            vals.get()[i] = CompilerUtils::constantOrThrow<double>(
                    currentValue, "elements of matrix literals must be parse-time constants"
            );
        }
        auto mat = DataObjectFactory::create<DenseMatrix<double>>(ctx->literal().size(), 1, vals);
        result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::MatrixConstantOp>(loc, utils.matrixOf(valueType),
                        builder.create<mlir::daphne::ConstantOp>(loc, reinterpret_cast<uint64_t>(mat))
                )
        );
    }
    else if(currentValue.getType().isSignlessInteger()){
        std::shared_ptr<bool[]> vals = std::shared_ptr<bool[]>(new bool[ctx->literal().size()]);
        for(unsigned i = 0; i < ctx->literal().size(); i++)
        {
            currentValue = utils.valueOrError(visitLiteral(ctx->literal(i)));
            if(currentValue.getType() != valueType)
                throw std::runtime_error("matrix of elements of different types");
            vals.get()[i] = CompilerUtils::constantOrThrow<bool>(
                    currentValue, "elements of matrix literals must be parse-time constants"
            );
        }
        auto mat = DataObjectFactory::create<DenseMatrix<bool>>(ctx->literal().size(), 1, vals);
        result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::MatrixConstantOp>(loc, utils.matrixOf(valueType),
                        builder.create<mlir::daphne::ConstantOp>(loc, reinterpret_cast<uint64_t>(mat))
                )
        );
    }
    else
        throw std::runtime_error("invalid value type for matrix literal");

    return result;
}

antlrcpp::Any DaphneDSLVisitor::visitIndexing(DaphneDSLGrammarParser::IndexingContext * ctx) {
    auto rows = ctx->rows
            ? visit(ctx->rows).as<std::pair<bool, antlrcpp::Any>>()
            : std::make_pair(false, antlrcpp::Any(nullptr));
    auto cols = ctx->cols
            ? visit(ctx->cols).as<std::pair<bool, antlrcpp::Any>>()
            : std::make_pair(false, antlrcpp::Any(nullptr));
    return std::make_pair(rows, cols);
}

antlrcpp::Any DaphneDSLVisitor::visitRange(DaphneDSLGrammarParser::RangeContext * ctx) {
    if(ctx->pos)
        return std::make_pair(true, antlrcpp::Any(utils.valueOrError(visit(ctx->pos))));
    else {
        mlir::Value posLowerIncl = ctx->posLowerIncl ? utils.valueOrError(visit(ctx->posLowerIncl)) : nullptr;
        mlir::Value posUpperExcl = ctx->posUpperExcl ? utils.valueOrError(visit(ctx->posUpperExcl)) : nullptr;
        return std::make_pair(
                posLowerIncl != nullptr || posUpperExcl != nullptr,
                antlrcpp::Any(std::make_pair(posLowerIncl, posUpperExcl))
        );
    }
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
                        loc, val
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
                        loc, val
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
                loc, val
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
    while(currOp) {
        auto nextOp = currOp->getNextNode();
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

void rectifyEarlyReturn(mlir::scf::IfOp ifOp) {
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
        if(!ifOp.getElseRegion().empty()) {
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
        ifOp.getCondition(),
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

            if(nested > levelOfMostNested) {
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
            rectifyEarlyReturn(ifOp);
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
        auto functionArguments = static_cast<std::vector<std::pair<std::string, mlir::Type>>>((visit(ctx->args)).as<std::vector<std::pair<std::string, mlir::Type>>>());
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
        auto blockArg = funcBlock->addArgument(std::get<1>(it), builder.getUnknownLoc());
        handleAssignmentPart(std::get<0>(it), nullptr, symbolTable, blockArg);
    }

    std::vector<mlir::Type> returnTypes;
    mlir::func::FuncOp functionOperation;
    if(ctx->retTys) {
        // early creation of FuncOp for recursion
        returnTypes = visit(ctx->retTys).as<std::vector<mlir::Type>>();
        functionOperation = createUserDefinedFuncOp(loc,
            builder.getFunctionType(funcArgTypes, returnTypes),
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
    // TODO Allow unknown type in return (could be subject to type inference).
    else if(returnOpTypes != returnTypes) {
        throw std::runtime_error(
            "Function `" + functionName + "` returns different type than specified in the definition");
    }
    functionOperation.getBody().push_front(funcBlock);

    symbolTable = globalSymbolTable;
    return functionOperation;
}

mlir::func::FuncOp DaphneDSLVisitor::createUserDefinedFuncOp(const mlir::Location &loc,
                                                       const mlir::FunctionType &funcType,
                                                       const std::string &functionName) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *moduleBody = module.getBody();
    auto functionSymbolName = utils.getUniqueFunctionSymbol(functionName);

    builder.setInsertionPoint(moduleBody, moduleBody->begin());
    auto functionOperation = builder.create<mlir::func::FuncOp>(loc, functionSymbolName, funcType);
    functionsSymbolMap.insert({functionName, functionOperation});
    return functionOperation;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArgs(DaphneDSLGrammarParser::FunctionArgsContext *ctx) {
    std::vector<std::pair<std::string, mlir::Type>> functionArguments;
    for(auto funcArgCtx: ctx->functionArg()) {
        functionArguments.push_back(visitFunctionArg(funcArgCtx).as<std::pair<std::string, mlir::Type>>());
    }
    return functionArguments;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArg(DaphneDSLGrammarParser::FunctionArgContext *ctx) {
    auto ty = utils.unknownType;
    if(ctx->ty) {
        ty = utils.typeOrError(visitFuncTypeDef(ctx->ty));
    }
    return std::make_pair(ctx->var->getText(), ty);
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionRetTypes(DaphneDSLGrammarParser::FunctionRetTypesContext *ctx) {
    std::vector<mlir::Type> retTys;
    for(auto ftdCtx : ctx->funcTypeDef())
        retTys.push_back(visitFuncTypeDef(ftdCtx).as<mlir::Type>());
    return retTys;
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
