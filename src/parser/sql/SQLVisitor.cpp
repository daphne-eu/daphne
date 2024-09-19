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

#include "antlr4-runtime.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include <ir/daphneir/Daphne.h>
#include <parser/sql/SQLVisitor.h>
#include <util/ErrorHandler.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <regex>

// ****************************************************************************
// Helper functions
// ****************************************************************************
/**
 * @brief Test if the flag at the position is set and returns result
 */
bool isBitSet(int64_t flag, int64_t position) {
    return ((flag >> position) & 1) == 1;
}

/**
 * @brief Sets the Flag at the given position with a value
 */
void setBit(int64_t &flag, int64_t position, int64_t val) {
    val = !!val;
    flag ^= (-val ^ flag) & (0b1 << position);
}

/**
 * @brief Flips the bit of the Flag at the position
 */
void toggleBit(int64_t &flag, int64_t position) {
    setBit(flag, position, !isBitSet(flag, position));
}

/**
 * @brief Creates a lower cast version of a string
 */
std::string toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

// ****************************************************************************
// Member Helper functions
// ****************************************************************************

template <typename T>
T fetch(std::unordered_map<std::string, T> x, const std::string &name) {
    auto search = x.find(name);
    return search != x.end() ? search->second : nullptr;
}

template <>
std::string fetch<std::string>(std::unordered_map<std::string, std::string> x,
                               const std::string &name) {
    auto search = x.find(name);
    if (search != x.end()) {
        return search->second;
    }
    return "";
}

void SQLVisitor::registerAlias(const std::string &name, mlir::Value arg) {
    alias[name] = arg;
}

std::string SQLVisitor::setFramePrefix(const std::string &framename,
                                       const std::string &prefix,
                                       bool necessary = true,
                                       bool ignore = false) {
    bool frameHasPrefix = !fetch<std::string>(framePrefix, framename).empty();
    if (frameHasPrefix) {
        if (necessary) {
            std::stringstream x;
            x << "Error: " << framename
              << " is marked as necessary for Prefix generation, but already "
                 "got a Prefix. Please consider an Alias\n";
            throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
        } else {
            return "";
        }
    }
    bool inuse =
        !fetch<std::string>(reverseFramePrefix, prefix).empty() && !ignore;
    std::string newPrefix = prefix;
    int i = 1;
    while (inuse) {
        std::stringstream x;
        x << prefix << i;
        newPrefix = x.str();
        inuse = !fetch<std::string>(reverseFramePrefix, newPrefix).empty();
    }

    if (!ignore) {
        reverseFramePrefix[newPrefix] = framename;
    }
    framePrefix[framename] = newPrefix;

    return newPrefix;
}

[[maybe_unused]] mlir::Value SQLVisitor::fetchAlias(const std::string &name) {
    auto res = fetch<mlir::Value>(alias, name);
    if (res != nullptr) {
        return res;
    }
    std::stringstream x;
    x << "Error: " << name << " does not name an Alias\n";
    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
}

mlir::Value SQLVisitor::fetchMLIR(const std::string &name) {
    mlir::Value res;
    res = fetch<mlir::Value>(alias, name);
    if (res != nullptr) {
        return res;
    }
    res = fetch<mlir::Value>(view, name);
    if (res != nullptr) {
        return res;
    }
    std::stringstream x;
    x << "Error: " << name
      << " was not registered with the Function \"registerView\" or were given "
         "an alias\n";
    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
}

std::string SQLVisitor::fetchPrefix(const std::string &name) {
    auto prefix = fetch<std::string>(framePrefix, name);
    return prefix.empty() ? "" : prefix;
}

bool SQLVisitor::hasMLIR(const std::string &name) {
    auto searchview = view.find(name);
    auto searchalias = alias.find(name);
    return (searchview != view.end() || searchalias != alias.end());
}

mlir::Value SQLVisitor::createStringConstant(std::string str) {
    // Replace escape sequences.
    str = std::regex_replace(str, std::regex(R"(\\b)"), "\b");
    str = std::regex_replace(str, std::regex(R"(\\f)"), "\f");
    str = std::regex_replace(str, std::regex(R"(\\n)"), "\n");
    str = std::regex_replace(str, std::regex(R"(\\r)"), "\r");
    str = std::regex_replace(str, std::regex(R"(\\t)"), "\t");
    str = std::regex_replace(str, std::regex(R"(\\\")"), "\"");
    str = std::regex_replace(str, std::regex(R"(\\\\)"), "\\");
    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(queryLoc, str));
}

mlir::Value SQLVisitor::castToMatrixColumn(mlir::Value toCast) {
    if (llvm::isa<mlir::daphne::MatrixType>(toCast.getType())) {
        return toCast;
    } else {
        mlir::Value numRow =
            static_cast<mlir::Value>(builder.create<mlir::daphne::NumRowsOp>(
                queryLoc, utils.sizeType, currentFrame));

        mlir::Value one =
            static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(
                queryLoc, static_cast<int64_t>(1)));

        return static_cast<mlir::Value>(builder.create<mlir::daphne::FillOp>(
            queryLoc, utils.matrixOf(toCast), toCast, numRow,
            utils.castSizeIf(one)));
    }
}

mlir::Value SQLVisitor::castToIntMatrixColumn(mlir::Value toCast) {
    mlir::Value toCastMatrix = castToMatrixColumn(toCast);
    mlir::Type vt = utils.getValueTypeByName("si64");
    mlir::Type resType = utils.matrixOf(vt);

    if (toCastMatrix.getType() != resType) {
        mlir::Value toCastFrame = matrixToFrame(
            toCastMatrix, "TempCol"); // We need this step because castOp can't
                                      // cast a matrix to a matrix

        return static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(
            queryLoc, resType, toCastFrame));
    }
    return toCastMatrix;
}

mlir::Value SQLVisitor::matrixToFrame(mlir::Value matrix,
                                      std::string newColumnName) {
    if (llvm::isa<mlir::daphne::MatrixType>(matrix.getType())) {
        // make a Frame from the Matrix.
        std::vector<mlir::Type> colTypes;
        std::vector<mlir::Value> cols;
        std::vector<mlir::Value> labels;

        colTypes.push_back(matrix.getType()
                               .dyn_cast<mlir::daphne::MatrixType>()
                               .getElementType());
        cols.push_back(matrix);
        labels.push_back(createStringConstant(newColumnName));

        mlir::Type t =
            mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::CreateFrameOp>(queryLoc, t, cols,
                                                        labels));
    } else {
        std::stringstream err_msg;
        err_msg << "matrixToFrame expects a mlir::daphne::MatrixType\n";
        throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                          err_msg.str());
    }
}

mlir::Value SQLVisitor::addMatrixToCurrentFrame(mlir::Value matrix,
                                                std::string newColumnName) {
    if (llvm::isa<mlir::daphne::MatrixType>(matrix.getType())) {
        mlir::Value add = matrixToFrame(matrix, newColumnName);

        // ADD new Frame to currentFrame
        std::vector<mlir::Type> currentFrame_colTypes;
        for (mlir::Type t : currentFrame.getType()
                                .dyn_cast<mlir::daphne::FrameType>()
                                .getColumnTypes())
            currentFrame_colTypes.push_back(t);
        for (mlir::Type t :
             add.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
            currentFrame_colTypes.push_back(t);
        mlir::Type resType = mlir::daphne::FrameType::get(
            builder.getContext(), currentFrame_colTypes);

        currentFrame =
            static_cast<mlir::Value>(builder.create<mlir::daphne::ColBindOp>(
                queryLoc, resType, currentFrame, add));
        return currentFrame;
    } else {
        std::stringstream err_msg;
        err_msg
            << "addMatrixToCurrentFrame expects a mlir::daphne::MatrixType\n";
        throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                          err_msg.str());
    }
}

mlir::Value SQLVisitor::getColIdx(mlir::Value frame, mlir::Value colName) {
    return static_cast<mlir::Value>(builder.create<mlir::daphne::GetColIdxOp>(
        queryLoc, utils.sizeType, frame, colName));
}

mlir::Value SQLVisitor::extractColumnAsMatrixFromFrame(mlir::Value frame,
                                                       mlir::Value colname) {

    // TODO: Integration of Transformation of ColName to ColIdx Op.
    // mlir::Value colIdx = getColIdx(frame, colname);
    mlir::Value col = extractColumnFromFrame(frame, colname);

    mlir::Type vt = utils.unknownType;
    mlir::Type resType = utils.matrixOf(vt);
    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::CastOp>(queryLoc, resType, col));
}

mlir::Attribute SQLVisitor::getGroupEnum(const std::string &func) {
    if (func == "count") {
        return static_cast<mlir::Attribute>(mlir::daphne::GroupEnumAttr::get(
            builder.getContext(), mlir::daphne::GroupEnum::COUNT));
    }
    if (func == "sum") {
        return static_cast<mlir::Attribute>(mlir::daphne::GroupEnumAttr::get(
            builder.getContext(), mlir::daphne::GroupEnum::SUM));
    }
    if (func == "min") {
        return static_cast<mlir::Attribute>(mlir::daphne::GroupEnumAttr::get(
            builder.getContext(), mlir::daphne::GroupEnum::MIN));
    }
    if (func == "max") {
        return static_cast<mlir::Attribute>(mlir::daphne::GroupEnumAttr::get(
            builder.getContext(), mlir::daphne::GroupEnum::MAX));
    }
    if (func == "avg") {
        return static_cast<mlir::Attribute>(mlir::daphne::GroupEnumAttr::get(
            builder.getContext(), mlir::daphne::GroupEnum::AVG));
    }
    std::stringstream x;
    x << "Error: " << func
      << " does not name an aggregation Function for Group\n";
    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
    return nullptr;
}

mlir::Attribute SQLVisitor::getCompareEnum(const std::string &op) {
    if (op == "=") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(), mlir::daphne::CompareOperation::Equal));
    }
    if (op == "<") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(),
                mlir::daphne::CompareOperation::LessThan));
    }
    if (op == "<=") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(),
                mlir::daphne::CompareOperation::LessEqual));
    }
    if (op == ">") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(),
                mlir::daphne::CompareOperation::GreaterThan));
    }
    if (op == ">=") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(),
                mlir::daphne::CompareOperation::GreaterEqual));
    }
    if (op == "<>" or op == "!=") {
        return static_cast<mlir::Attribute>(
            mlir::daphne::CompareOperationAttr::get(
                builder.getContext(),
                mlir::daphne::CompareOperation::NotEqual));
    }
    std::stringstream x;
    x << "Error: " << op << " does not name a compare operation\n";
    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
}

std::string SQLVisitor::getEnumLabelExt(const std::string &func) {
    return mlir::daphne::stringifyGroupEnum(
               getGroupEnum(func)
                   .dyn_cast<mlir::daphne::GroupEnumAttr>()
                   .getValue())
        .str();
}

mlir::Value SQLVisitor::extractColumnFromFrame(mlir::Value frame,
                                               mlir::Value columnName) {
    mlir::Type vt = utils.unknownType;
    mlir::Type resTypeCol =
        mlir::daphne::FrameType::get(builder.getContext(), {vt});

    return (static_cast<mlir::Value>(builder.create<mlir::daphne::ExtractColOp>(
        queryLoc, resTypeCol, frame, columnName)));
}

// ****************************************************************************
// Visitor functions wrongeedsda
// ****************************************************************************

// script
antlrcpp::Any SQLVisitor::visitScript(SQLGrammarParser::ScriptContext *ctx) {
    mlir::Value res =
        utils.valueOrError(utils.getLoc(ctx->start), visitChildren(ctx));
    return res;
}

// sql
antlrcpp::Any SQLVisitor::visitSql(SQLGrammarParser::SqlContext *ctx) {
    mlir::Value res = valueOrErrorOnVisit(ctx->query());
    return res;
}

// query
antlrcpp::Any SQLVisitor::visitQuery(SQLGrammarParser::QueryContext *ctx) {
    mlir::Value res = valueOrErrorOnVisit(ctx->select());
    return res;
}

// select
antlrcpp::Any SQLVisitor::visitSelect(SQLGrammarParser::SelectContext *ctx) {
    mlir::Value res;

    // Setting Codegeneration for Where Clause
    setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);

    // Creating a Frame using FROM and JOIN
    try {
        currentFrame = valueOrErrorOnVisit(ctx->tableExpr());
    } catch (std::runtime_error &e) {
        std::stringstream err_msg;
        err_msg << "Error during From statement. "
                << "Couldn't create Frame.\n\t\t" << e.what();
        throw ErrorHandler::rethrowError("SQLVisitor (visitSelect)",
                                         err_msg.str());
    }

    // If a where clause exist, filter <currentFrame> accordingly.
    if (ctx->whereClause()) {
        currentFrame = valueOrErrorOnVisit(ctx->whereClause());
    }

    // In case of a group by clause, we deactivate code generation for a moment
    // to scan the projection and havingClause for identifiers that need to be
    // included in the group. NOTE: in case the having or projection includes an
    // aggregation function we are going to generate the code on which the
    // aggregation and extend the currentFrame with it.
    if (ctx->groupByClause()) {
        setBit(sqlFlag, (int64_t)SQLBit::group, 1);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 0);
        visit(ctx->groupByClause());
        for (size_t i = 0; i < ctx->selectExpr().size(); i++) {
            visit(ctx->selectExpr(i));
        }
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);
        visit(ctx->groupByClause());
    }

    if (ctx->orderByClause()) {
        currentFrame = valueOrErrorOnVisit(ctx->orderByClause());
    }

    // Runs over the projections and seeks columns and adds them to a Frame,
    // which is the result of this function
    res = valueOrErrorOnVisit(ctx->selectExpr(0));
    for (auto i = 1ul; i < ctx->selectExpr().size(); i++) {
        mlir::Value add;
        try {
            add = valueOrErrorOnVisit(ctx->selectExpr(i));
        } catch (std::runtime_error &e) {
            std::stringstream err_msg;
            err_msg << "Something went wrong in SelectExpr.\n\t\t" << e.what();
            throw ErrorHandler::rethrowError("SQLVisitor (visitSelect)",
                                             err_msg.str());
        }
        try {
            std::vector<mlir::Type> colTypes;
            for (mlir::Type t : res.getType()
                                    .dyn_cast<mlir::daphne::FrameType>()
                                    .getColumnTypes())
                colTypes.push_back(t);
            for (mlir::Type t : add.getType()
                                    .dyn_cast<mlir::daphne::FrameType>()
                                    .getColumnTypes())
                colTypes.push_back(t);
            mlir::Type resType =
                mlir::daphne::FrameType::get(builder.getContext(), colTypes);

            res = static_cast<mlir::Value>(
                builder.create<mlir::daphne::ColBindOp>(queryLoc, resType, res,
                                                        add));
        } catch (std::runtime_error &e) {
            std::stringstream err_msg;
            err_msg << "Error during SELECT statement. "
                    << "Couldn't Insert Extracted Columns.\n\t\t" << e.what();
            throw ErrorHandler::rethrowError("SQLVisitor (visitSelect)",
                                             err_msg.str());
        }
    }
    currentFrame = res;
    if (ctx->distinctExpr()) {
        res = valueOrErrorOnVisit(ctx->distinctExpr());
    }
    return res;
}

// subquery
antlrcpp::Any
SQLVisitor::visitSubquery(SQLGrammarParser::SubqueryContext *ctx) {
    // TODO: Subquery Implementations
    return visitChildren(ctx);
}

// subqueryExpr
antlrcpp::Any
SQLVisitor::visitSubqueryExpr(SQLGrammarParser::SubqueryExprContext *ctx) {
    // TODO: Subquery Implementations
    return nullptr;
}

// SelectExpr
antlrcpp::Any
SQLVisitor::visitSelectExpr(SQLGrammarParser::SelectExprContext *ctx) {
    mlir::Value matrix;
    antlrcpp::Any vExpr = visit(ctx->var);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    // we get a Matrix or int/float value. From this we generate a Matrix.
    mlir::Value expr = utils.valueOrError(utils.getLoc(ctx->var->start), vExpr);

    if (llvm::isa<mlir::daphne::FrameType>(expr.getType())) {
        return expr;
    }

    matrix = castToMatrixColumn(expr);

    // Now we look up what the label for the result should be
    std::string label;
    if (ctx->aka) {
        label = ctx->aka->getText();
    } else {
        label = ctx->getText();
    }
    // And generate with the matrix and label a Frame which we return
    return matrixToFrame(matrix, label);
}

// tableExpr
antlrcpp::Any
SQLVisitor::visitTableExpr(SQLGrammarParser::TableExprContext *ctx) {
    // We set the current frame as the result of the fromExpr
    currentFrame = valueOrErrorOnVisit(ctx->fromExpr());
    // And join other frames to the currentFrame.
    for (size_t i = 0; i < ctx->joinExpr().size(); i++) {
        currentFrame = valueOrErrorOnVisit(ctx->joinExpr(i));
    }
    return currentFrame;
}

// distinctExpr
antlrcpp::Any
SQLVisitor::visitDistinctExpr(SQLGrammarParser::DistinctExprContext *ctx) {
    if (isBitSet(sqlFlag, (int64_t)SQLBit::group) // If group is active
        && columnName.size())                     // AND there is an aggregation
    {
        throw ErrorHandler::compilerError(
            queryLoc, "SQLVisitor",
            "DISTINCT with GROUP BY and Aggregation is not supported");
    } else if (isBitSet(sqlFlag, (int64_t)SQLBit::group) // If group is active
               || columnName.size()) // OR there is an aggregation
    {
        return currentFrame; // due to earlier grouping/aggregation the result
                             // is already distinct
    }

    mlir::Value starLiteral = createStringConstant("*");
    std::vector<mlir::Value> cols{starLiteral};
    std::vector<mlir::Value> aggs;
    std::vector<mlir::Attribute> functions;
    mlir::Type vt = utils.unknownType;
    std::vector<mlir::Type> colTypes{vt};
    mlir::Type resType =
        mlir::daphne::FrameType::get(builder.getContext(), colTypes);
    return static_cast<mlir::Value>(builder.create<mlir::daphne::GroupOp>(
        queryLoc, resType, currentFrame, cols, aggs,
        builder.getArrayAttr(functions)));
}

// fromExpr
antlrcpp::Any SQLVisitor::visitTableIdentifierExpr(
    SQLGrammarParser::TableIdentifierExprContext *ctx) {
    try {
        mlir::Value var = valueOrErrorOnVisit(ctx->var);
        return var;
    } catch (std::runtime_error &e) {
        throw ErrorHandler::compilerError(
            queryLoc, "SQLVisitor (visitTableIdentifierExpr)", e.what());
    }
}

antlrcpp::Any
SQLVisitor::visitCartesianExpr(SQLGrammarParser::CartesianExprContext *ctx) {
    // we have to at least two frames in the fromExpr. We join them together
    // with the Cartesian product.
    try {
        mlir::Location loc = utils.getLoc(ctx->start);
        mlir::Value res;
        mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
        mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

        std::vector<mlir::Type> colTypes;
        for (mlir::Type t : lhs.getType()
                                .dyn_cast<mlir::daphne::FrameType>()
                                .getColumnTypes()) {
            colTypes.push_back(t);
        }
        for (mlir::Type t : rhs.getType()
                                .dyn_cast<mlir::daphne::FrameType>()
                                .getColumnTypes()) {
            colTypes.push_back(t);
        }
        mlir::Type t =
            mlir::daphne::FrameType::get(builder.getContext(), colTypes);

        res = static_cast<mlir::Value>(
            builder.create<mlir::daphne::CartesianOp>(loc, t, lhs, rhs));
        return res;
    } catch (std::runtime_error &) {
        throw ErrorHandler::rethrowError(
            "SQLVisitor (visitCartesianExpr)",
            "Unexpected Error during Cartesian operation");
    }
}

// joinExpr
antlrcpp::Any
SQLVisitor::visitInnerJoin(SQLGrammarParser::InnerJoinContext *ctx) {
    // we join to frames together. One is the currentFrame and the other is a
    // new frame. The argument that referneces the currentFrame has to be on the
    // left side of the Comparisons and the to be joined on the right side.
    // This behavior could be changed here.
    // TODO: Make the position independent
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value tojoin = valueOrErrorOnVisit(ctx->var);

    std::vector<mlir::Type> colTypes;
    for (mlir::Type t : currentFrame.getType()
                            .dyn_cast<mlir::daphne::FrameType>()
                            .getColumnTypes())
        colTypes.push_back(t);
    for (mlir::Type t :
         tojoin.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

    // ctx->CMP_OP(0)->getText()
    if (ctx->op->getText() == "=" && ctx->selectIdent().size() == 2) {
        // rhs is join
        // lhs is currentFrame
        mlir::Value rhsName = valueOrErrorOnVisit(ctx->rhs);
        mlir::Value lhsName = valueOrErrorOnVisit(ctx->lhs);

        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::InnerJoinOp>(
                loc, t, currentFrame, tojoin, rhsName, lhsName));
    }

    std::vector<mlir::Value> rhsNames;
    std::vector<mlir::Value> lhsNames;
    std::vector<mlir::Attribute> ops;

    for (auto i = 0ul; i < ctx->selectIdent().size() / 2; i++) {
        mlir::Value lhsName = valueOrErrorOnVisit(ctx->selectIdent(i * 2));
        mlir::Value rhsName = valueOrErrorOnVisit(ctx->selectIdent(i * 2 + 1));
        mlir::Attribute op = getCompareEnum(ctx->CMP_OP(i)->getText());

        lhsNames.push_back(lhsName);
        rhsNames.push_back(rhsName);
        ops.push_back(op);
    }

    return static_cast<mlir::Value>(builder.create<mlir::daphne::ThetaJoinOp>(
        loc, t, currentFrame, tojoin, lhsNames, rhsNames,
        builder.getArrayAttr(ops)));
}

antlrcpp::Any
SQLVisitor::visitWhereClause(SQLGrammarParser::WhereClauseContext *ctx) {
    // Creates a FilterRowOp with the result of a generalExpr. The result is a
    // matrix or a single value, expr. expr gets cast to a matrix, which
    // FilterRowOp uses. IMPORTANT: FilterRowOp takes up the work to make a
    // int/float into a boolean for the filtering.
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value filter;

    mlir::Value expr = valueOrErrorOnVisit(ctx->cond);
    filter = castToMatrixColumn(expr);

    mlir::Value v =
        static_cast<mlir::Value>(builder.create<mlir::daphne::FilterRowOp>(
            loc, currentFrame.getType(), currentFrame, filter));
    return v;
}

// groupByClause
antlrcpp::Any
SQLVisitor::visitGroupByClause(SQLGrammarParser::GroupByClauseContext *ctx) {
    // groupByClause has two moods:
    // Codegeneration = false:
    //   groupByClause collects all the column names by which a grouping
    //   should occur.
    //   TODO: call the havingClause.
    // Codegeneration = true:
    //   groupByClause creates the groupingOperation with the gathered
    //   information from here, having and the projections.
    //   followed by a having check on the newly grouped result.
    mlir::Location loc = utils.getLoc(ctx->start);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        for (size_t i = 0; i < ctx->selectIdent().size(); i++) {
            groupName.push_back(valueOrErrorOnVisit(ctx->selectIdent(i)));
            grouped[ctx->selectIdent(i)->getText()] = 1;
        }
        if (ctx->havingClause()) {
            visit(ctx->havingClause());
        }
        return nullptr;
    } else {
        mlir::Type vt = utils.unknownType;
        std::vector<mlir::Type> colTypes;
        for (size_t i = 0; i < groupName.size() + columnName.size(); i++) {
            colTypes.push_back(vt);
        }
        mlir::Type resType =
            mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        currentFrame =
            static_cast<mlir::Value>(builder.create<mlir::daphne::GroupOp>(
                loc, resType, currentFrame, groupName, columnName,
                builder.getArrayAttr(functionName)));
        if (ctx->havingClause()) {
            currentFrame = valueOrErrorOnVisit(ctx->havingClause());
        }
    }
    return nullptr;
}

// havingClause
antlrcpp::Any
SQLVisitor::visitHavingClause(SQLGrammarParser::HavingClauseContext *ctx) {
    // Same as Where
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value filter;
    antlrcpp::Any vExpr = visit(ctx->cond);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value expr =
        utils.valueOrError(utils.getLoc(ctx->cond->start), vExpr);
    filter = castToMatrixColumn(expr);

    mlir::Value v =
        static_cast<mlir::Value>(builder.create<mlir::daphne::FilterRowOp>(
            loc, currentFrame.getType(), currentFrame, filter));
    return v;
}

antlrcpp::Any
SQLVisitor::visitOrderByClause(SQLGrammarParser::OrderByClauseContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    std::vector<mlir::Value> columnIdxs;
    std::vector<mlir::Value> asc;
    for (auto i = 0ul; i < ctx->selectIdent().size(); i++) {
        mlir::Value boolean = valueOrErrorOnVisit(ctx->orderInformation(i));
        mlir::Value columnName = valueOrErrorOnVisit(ctx->selectIdent(i));
        mlir::Value idx = getColIdx(currentFrame, columnName);
        columnIdxs.push_back(utils.castSizeIf(idx));
        asc.push_back(utils.castBoolIf(boolean));
    }
    mlir::Value returnFrame = static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(loc, false));
    return static_cast<mlir::Value>(builder.create<mlir::daphne::OrderOp>(
        loc, currentFrame.getType().dyn_cast<mlir::daphne::FrameType>(),
        currentFrame, columnIdxs, asc, returnFrame));
}

antlrcpp::Any SQLVisitor::visitOrderInformation(
    SQLGrammarParser::OrderInformationContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    // ASC is default
    if (ctx->desc) {
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, false));
    }
    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(loc, true));
}
// generalExpr

// For the following generalExpr:
//   If Code generation is turned off, it will still visit the generalExpr
//       underneath and then return a nullptr.
//   If Code generation is turned on, it will generate Code like an addition.
//   If something else is happening, it got additional Documentation.

antlrcpp::Any
SQLVisitor::visitLiteralExpr(SQLGrammarParser::LiteralExprContext *ctx) {
    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }
    return valueOrErrorOnVisit(ctx->literal());
}

antlrcpp::Any
SQLVisitor::visitIdentifierExpr(SQLGrammarParser::IdentifierExprContext *ctx) {
    if (isBitSet(sqlFlag, (int64_t)SQLBit::group) // If group is active
        && !isBitSet(sqlFlag,
                     (int64_t)SQLBit::agg) // AND there isn't an aggregation
        && grouped.count(ctx->selectIdent()->getText()) ==
               0 // AND the label is not in group expr
        && ctx->selectIdent()
                   ->getText()[ctx->selectIdent()->getText().length() - 1] !=
               '*' // AND the label does not end in * (* or f.*)
        && grouped.count("*") == 0) // AND there is no * in group expr
    {
        std::stringstream err_msg;
        err_msg << "Error during a generalExpr. \""
                << ctx->selectIdent()->getText() << "\" Must be part of "
                << "the Group Expression or have an Aggregation Function";
        throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                          err_msg.str());
    }

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    auto label = ctx->selectIdent()->getText();
    if (label.compare("*") == 0) { // SELECT *
        return valueOrErrorOnVisit(ctx->selectIdent());
    } else if (label.compare(label.length() - 2, 2, ".*") ==
               0) { // SELECT frame.*
        mlir::Value colname = valueOrErrorOnVisit(ctx->selectIdent());
        return extractColumnFromFrame(currentFrame, colname);
    }

    mlir::Value colname = valueOrErrorOnVisit(ctx->selectIdent());
    return extractColumnAsMatrixFromFrame(currentFrame, colname);
}

antlrcpp::Any
SQLVisitor::visitStarExpr(SQLGrammarParser::StarExprContext *ctx) {
    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    } else if (isBitSet(sqlFlag, (int64_t)SQLBit::group) // If group is active
               && !groundGroupColumns.empty() // AND there is an aggregation
               && isBitSet(sqlFlag,
                           (int64_t)SQLBit::codegen)) { // AND codegen is active
        std::string columnName = groundGroupColumns.begin()->c_str();
        groundGroupColumns.erase(groundGroupColumns.begin());

        mlir::Value colname = createStringConstant(columnName);

        mlir::Value resultFrame = extractColumnFromFrame(currentFrame, colname);
        std::set<std::string>::iterator itr;
        for (itr = groundGroupColumns.begin(); itr != groundGroupColumns.end();
             itr++) {
            mlir::Value groupColname = createStringConstant(itr->c_str());
            mlir::Value addFrame =
                extractColumnFromFrame(currentFrame, groupColname);

            std::vector<mlir::Type> colTypes;
            for (mlir::Type t : resultFrame.getType()
                                    .dyn_cast<mlir::daphne::FrameType>()
                                    .getColumnTypes())
                colTypes.push_back(t);
            for (mlir::Type t : addFrame.getType()
                                    .dyn_cast<mlir::daphne::FrameType>()
                                    .getColumnTypes())
                colTypes.push_back(t);
            mlir::Type resType =
                mlir::daphne::FrameType::get(builder.getContext(), colTypes);
            resultFrame = static_cast<mlir::Value>(
                builder.create<mlir::daphne::ColBindOp>(queryLoc, resType,
                                                        resultFrame, addFrame));
        }
        return resultFrame;
    } else if (!isBitSet(sqlFlag,
                         (int64_t)SQLBit::group) // If group is not active
               && isBitSet(sqlFlag,
                           (int64_t)SQLBit::agg) // AND there is an aggreagtion
               &&
               isBitSet(sqlFlag,
                        (int64_t)SQLBit::codegen)) { // AND codegen is active)
        throw ErrorHandler::compilerError(
            queryLoc, "SQLVisitor",
            "Using the asterisk with only aggregation functions is not "
            "allowed");
    } else {
        return currentFrame;
    }
}

antlrcpp::Any
SQLVisitor::visitGroupAggExpr(SQLGrammarParser::GroupAggExprContext *ctx) {
    // This function should only be called if there is a "group by" in the query
    // Codegeneration = false:
    //   This function activates the code generation and ignores the aggreagtion
    //   It lets the generalExpr create code as usual. When a Value is returned
    //   it takes this value and adds it to the currentFrame under a new and
    //   somewhat unique name.
    //   (TODO: there might be an issue with not unique name generation)
    //   the name gets saved alongside with the aggregation function.
    //   After all that, code generation is turned off again.
    // Codegeneration = true:
    //   The function looks up the unique name again and extracts a matrix from
    //   the currentFrame. This Matrix is the result of this function.
    std::string newColumnName =
        "group_" + std::to_string(groupCounter) + "_" + ctx->var->getText();
    // Increment groupCounter
    groupCounter++;

    groundGroupColumns.insert(ctx->var->getText());

    if (ctx->var->getText()[ctx->var->getText().length() - 1] == '*') {
        throw ErrorHandler::compilerError(
            queryLoc, "SQLVisitor",
            "Using the asterisk in aggregations is not allowed");
    }

    // Run aggreagation for whole column
    if (!isBitSet(sqlFlag, (int64_t)SQLBit::group) &&
        isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        mlir::Location loc = utils.getLoc(ctx->start);

        mlir::Value col = valueOrErrorOnVisit(ctx->var);

        mlir::Type resTypeCol =
            col.getType().dyn_cast<mlir::daphne::MatrixType>().getElementType();

        std::string func = toLower(ctx->func->getText());

        mlir::Value result;
        if (func == "count") {
            result = utils.castSI64If(static_cast<mlir::Value>(
                builder.create<mlir::daphne::NumRowsOp>(loc, utils.sizeType,
                                                        col)));
        }
        if (func == "sum") {
            result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::AllAggSumOp>(loc, resTypeCol,
                                                          col));
        }
        if (func == "min") {
            result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::AllAggMinOp>(loc, resTypeCol,
                                                          col));
        }
        if (func == "max") {
            result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::AllAggMaxOp>(loc, resTypeCol,
                                                          col));
        }
        if (func == "avg") {
            result = static_cast<mlir::Value>(
                builder.create<mlir::daphne::AllAggMeanOp>(loc, resTypeCol,
                                                           col));
        }

        std::string newColumnNameAppended =
            getEnumLabelExt(func) + "(" + newColumnName + ")";

        return utils.castIf(utils.matrixOf(result), result);

        std::stringstream x;
        x << "Error: " << func
          << " does not name a supported aggregation function";
        throw ErrorHandler::compilerError(queryLoc, "SQLVisitor", x.str());
    }

    if (isBitSet(sqlFlag,
                 (int64_t)SQLBit::agg)) { // Not allowed nested Function Call
        throw ErrorHandler::compilerError(
            queryLoc, "SQLVisitor", "Nested Aggregation Functions not allowed");
    }

    // create Column pre Group for in group Aggregation
    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        columnName.push_back(createStringConstant(newColumnName));
        const std::string &func = toLower(ctx->func->getText());
        functionName.push_back(getGroupEnum(func));

        setBit(sqlFlag, (int64_t)SQLBit::agg, 1);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);
        mlir::Value expr = valueOrErrorOnVisit(ctx->generalExpr());
        setBit(sqlFlag, (int64_t)SQLBit::agg, 0);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 0);

        mlir::Value matrix = castToMatrixColumn(expr);
        currentFrame = addMatrixToCurrentFrame(matrix, newColumnName);
        return nullptr;
    } else { // Get Column after Group
        std::string newColumnName = "group_" +
                                    std::to_string(groupCounterCodegen) + "_" +
                                    ctx->var->getText();
        // Increment groupCounter
        groupCounterCodegen++;
        const std::string &func = toLower(ctx->func->getText());
        std::string newColumnNameAppended =
            getEnumLabelExt(func) + "(" + newColumnName + ")";
        mlir::Value colname = createStringConstant(newColumnNameAppended);
        return extractColumnAsMatrixFromFrame(currentFrame,
                                              colname); // returns Matrix
    }
}

antlrcpp::Any SQLVisitor::visitParanthesesExpr(
    SQLGrammarParser::ParanthesesExprContext *ctx) {
    antlrcpp::Any vRes = visit(ctx->generalExpr());
    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }
    return utils.valueOrError(utils.getLoc(ctx->generalExpr()->start), vRes);
}

antlrcpp::Any SQLVisitor::visitMulExpr(SQLGrammarParser::MulExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(utils.getLoc(ctx->lhs->start), vLhs);
    mlir::Value rhs = utils.valueOrError(utils.getLoc(ctx->rhs->start), vRhs);

    if (op == "*")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwMulOp>(loc, lhs, rhs));
    if (op == "/")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwDivOp>(loc, lhs, rhs));

    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                      "unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitAddExpr(SQLGrammarParser::AddExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(utils.getLoc(ctx->lhs->start), vLhs);
    mlir::Value rhs = utils.valueOrError(utils.getLoc(ctx->rhs->start), vRhs);

    if (op == "+")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwAddOp>(loc, lhs, rhs));
    if (op == "-")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwSubOp>(loc, lhs, rhs));

    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                      "unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitCmpExpr(SQLGrammarParser::CmpExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(utils.getLoc(ctx->lhs->start), vLhs);
    mlir::Value rhs = utils.valueOrError(utils.getLoc(ctx->rhs->start), vRhs);

    if (op == "=")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwEqOp>(loc, lhs, rhs));
    if (op == "<>")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwNeqOp>(loc, lhs, rhs));
    if (op == "<")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwLtOp>(loc, lhs, rhs));
    if (op == "<=")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwLeOp>(loc, lhs, rhs));
    if (op == ">")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwGtOp>(loc, lhs, rhs));
    if (op == ">=")
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::EwGeOp>(loc, lhs, rhs));

    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                      "unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitAndExpr(SQLGrammarParser::AndExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(utils.getLoc(ctx->lhs->start), vLhs);
    mlir::Value rhs = utils.valueOrError(utils.getLoc(ctx->rhs->start), vRhs);

    lhs = castToIntMatrixColumn(lhs);
    rhs = castToIntMatrixColumn(rhs);

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::EwAndOp>(loc, lhs, rhs));
}

antlrcpp::Any SQLVisitor::visitOrExpr(SQLGrammarParser::OrExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if (!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)) {
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(utils.getLoc(ctx->lhs->start), vLhs);
    mlir::Value rhs = utils.valueOrError(utils.getLoc(ctx->rhs->start), vRhs);

    lhs = castToIntMatrixColumn(lhs);
    rhs = castToIntMatrixColumn(rhs);

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::EwOrOp>(loc, lhs, rhs));
}

// tableReference
//  Returns a modified Frame.
//  Modification: the Frame labels get a prefix.
antlrcpp::Any
SQLVisitor::visitTableReference(SQLGrammarParser::TableReferenceContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    std::string var = ctx->var->getText();
    std::string prefix = ctx->var->getText();
    try {
        mlir::Value res = fetchMLIR(var);
        registerAlias(var, res);
        if (ctx->aka) {
            std::string aka = ctx->aka->getText();
            registerAlias(aka, res);
            prefix = setFramePrefix(aka, aka);
        }

        setFramePrefix(var, prefix, !ctx->aka, ctx->aka);

        mlir::Value prefixSSA = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, prefix));

        res = static_cast<mlir::Value>(
            builder.create<mlir::daphne::SetColLabelsPrefixOp>(
                loc,
                res.getType()
                    .dyn_cast<mlir::daphne::FrameType>()
                    .withSameColumnTypes(),
                res, prefixSSA));
        return res;
    } catch (std::runtime_error &) {
        throw ErrorHandler::rethrowError("SQLVisitor (visitTableReference)",
                                         "Frame " + var +
                                             " referenced before assignment");
    }
}

// selectIdent //rowReference

// Returns A SSA to StringLabel for an ExtractColOp
// If a Frame is referenced, it checks its availability.
antlrcpp::Any
SQLVisitor::visitStringIdent(SQLGrammarParser::StringIdentContext *ctx) {
    std::string getSTR;
    std::string columnSTR = ctx->var->getText();
    std::string frameSTR;

    if (ctx->frame) {
        if (!hasMLIR(ctx->frame->getText())) {
            throw ErrorHandler::compilerError(
                queryLoc, "SQLVisitor",
                "Unknown Frame: " + ctx->frame->getText() +
                    " use before declaration during selection");
        }
        std::string framePrefix_ = fetchPrefix(ctx->frame->getText());
        if (!framePrefix_.empty()) {
            frameSTR = framePrefix_ + ".";
        } else {
            frameSTR = "";
        }
        getSTR = frameSTR + columnSTR;
    } else {
        getSTR = columnSTR;
    }
    return createStringConstant(getSTR);
}

// literal
antlrcpp::Any SQLVisitor::visitLiteral(SQLGrammarParser::LiteralContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    if (auto lit = ctx->INT_LITERAL()) {
        // ToDo: converted from atol to stol for safety -> check perf
        int64_t val = std::stol(lit->getText());
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, val));
    }
    if (auto lit = ctx->FLOAT_LITERAL()) {
        // ToDo: converted from atof to std::stod for safety -> check perf
        double val = std::stod(lit->getText());
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, val));
    }
    throw ErrorHandler::compilerError(queryLoc, "SQLVisitor",
                                      "unexpected literal");
}
