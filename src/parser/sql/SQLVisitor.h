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

#pragma once

#include <parser/ParserUtils.h>
#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
#include "SQLGrammarParser.h"
#include "SQLGrammarVisitor.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

class SQLVisitor : public SQLGrammarVisitor {

    ParserUtils utils;
    mlir::OpBuilder builder;

//special Variables
    mlir::Value currentFrame; //holds the complete Frame with all columns
    mlir::Location queryLoc;


//Helper Functions:

    /**
     * @brief creates a mlir-string-value from a c++ String
     */
    mlir::Value createStringConstant(std::string str);

    /**
     * @brief casts a single Attribute Value to a Matrix. Save for Matrix Input.
     */
    mlir::Value castToMatrixColumn(mlir::Value toCast);
    mlir::Value castToMatrixColumnWithOneEntry(mlir::Value toCast);
    mlir::Value castToIntMatrixColumn(mlir::Value toCast);
    /**
     * @brief Creates a Frame out of a Matrix Column and a name
     */
    mlir::Value matrixToFrame(
        mlir::Value matrix, std::string newColumnName);

    /**
     * @brief creates ColBindOp to add the matirx to the currentFrame.
     */
    mlir::Value addMatrixToCurrentFrame(
        mlir::Value matrix, std::string newColumnName);

    /**
    * @brief creates a GetColIdxOp for a specific colName of the frame.
    */
    mlir::Value getColIdx(
        mlir::Value frame, mlir::Value colName);

    /**
     * @brief creates ExtractColOp and CastOp
     */
    mlir::Value extractColumnAsMatrixFromFrame(
        mlir::Value frame, mlir::Value colname);

    /**
     * @brief returns GroupEnumAttr for a given aggregation function
     *
     * TODO: extend if more aggregation functions get implemented.
     */
    mlir::Attribute getGroupEnum(const std::string& func);

    /**
     * @brief returns CompareEnumAttr for a given compare operation
     */
    mlir::Attribute getCompareEnum(const std::string& op);

    /**
     * @brief returns result of stringifyGroupEnum for the given func.
     */
    std::string getEnumLabelExt(const std::string& func);

    /**
     * @brief returns a frame in which the contents of a column specified by
     * the columnName is copied.
     */
    mlir::Value extractColumnFromFrame(mlir::Value frame, mlir::Value columnName);


//Data Structures and access functions
    std::unordered_map <std::string, mlir::Value> view; //name, mlir::Value
    std::unordered_map <std::string, mlir::Value> alias; //name, mlir::Value

    std::unordered_map <std::string, std::string> framePrefix; //framename, prefix
    std::unordered_map<std::string, std::string> reverseFramePrefix; //prefix, framename


    /**
     * @brief adds a mlir Value under the string into the alias map for later lookup.
     */
    void registerAlias(const std::string& framename, mlir::Value arg);

    /**
     * @brief first looks up the Alias map and if the item is not found it looks
     * into the View map.
     */
    mlir::Value fetchMLIR(const std::string& framename);

    /**
     * @brief look up in Alias map
     */
    [[maybe_unused]] mlir::Value fetchAlias(const std::string& framename);

    /**
     * @brief looks up if the string specifies a mlir Value in the Alias or View Map
     */
    bool hasMLIR(const std::string& name);

    /**
     * @brief checks if a given prefix already given to annother framename otherwise
     * registers the prefix for this framename
     */
    std::string setFramePrefix(const std::string& framename, const std::string& prefix, bool necessary, bool ignore);

    /**
     * @brief looks up the prefix for a given framename
     */
    std::string fetchPrefix(const std::string& framename);

    //TODO: Recognize Literals and somehow handle them for the group expr.
//GROUP Information
    std::unordered_map <std::string, int8_t> grouped;
    std::vector<mlir::Value> groupName;
    std::vector<mlir::Value> columnName;
    std::vector<mlir::Attribute> functionName;
    std::set<std::string> groundGroupColumns;

    //Flags
    enum class SQLBit{group=0, codegen, agg, checkgroup};
    //group has group clause, activated codegen, is a complex general Expression, is a complex Group Expression, has aggregation function.
    int64_t sqlFlag = 0;

    //Counter for the group names, to enable multiple aggregations on the same column and to avoid name clashes.
    //We need two counter, as we need to reproduce the same names when actually doing the code generation.
    int64_t groupCounter = 0;
    int64_t groupCounterCodegen = 0;

public:
  [[maybe_unused]] explicit SQLVisitor(mlir::OpBuilder &builder,
                                       mlir::daphne::SqlOp sqlOp)
      : utils(builder), builder(builder), queryLoc(sqlOp.getLoc()){};

  SQLVisitor(mlir::OpBuilder &builder,
             std::unordered_map<std::string, mlir::Value> view_arg,
             mlir::daphne::SqlOp sqlOp)
      : utils(builder), builder(builder), queryLoc(sqlOp.getLoc()) {
      view = std::move(view_arg);
  };

//script
    antlrcpp::Any visitScript(SQLGrammarParser::ScriptContext * ctx) override;

//sql
    antlrcpp::Any visitSql(SQLGrammarParser::SqlContext * ctx) override;

//query
    antlrcpp::Any visitQuery(SQLGrammarParser::QueryContext * ctx) override;

//select
    antlrcpp::Any visitSelect(SQLGrammarParser::SelectContext * ctx) override;

//subquery
    antlrcpp::Any visitSubquery(SQLGrammarParser::SubqueryContext * ctx) override;

//subqueryExpr
    antlrcpp::Any visitSubqueryExpr(SQLGrammarParser::SubqueryExprContext * ctx) override;

//selectExpr
    antlrcpp::Any visitSelectExpr(SQLGrammarParser::SelectExprContext * ctx) override;

//tableExpr
    antlrcpp::Any visitTableExpr(SQLGrammarParser::TableExprContext * ctx) override;

//distinctExpr
    antlrcpp::Any visitDistinctExpr(SQLGrammarParser::DistinctExprContext * ctx) override;

//fromExpr
    antlrcpp::Any visitTableIdentifierExpr(SQLGrammarParser::TableIdentifierExprContext *ctx) override;

    antlrcpp::Any visitCartesianExpr(SQLGrammarParser::CartesianExprContext * ctx) override;

//joinExpr
    antlrcpp::Any visitInnerJoin(SQLGrammarParser::InnerJoinContext * ctx) override;

//whereClause
    antlrcpp::Any visitWhereClause(SQLGrammarParser::WhereClauseContext * ctx) override;

//groupByClause
    antlrcpp::Any visitGroupByClause(SQLGrammarParser::GroupByClauseContext * ctx) override;

//havingClause
    antlrcpp::Any visitHavingClause(SQLGrammarParser::HavingClauseContext * ctx) override;

//orderByClause
    antlrcpp::Any visitOrderByClause(SQLGrammarParser::OrderByClauseContext * ctx) override;

//orderInformation
    antlrcpp::Any visitOrderInformation(SQLGrammarParser::OrderInformationContext * ctx) override;

//generalExpr
    antlrcpp::Any visitLiteralExpr(SQLGrammarParser::LiteralExprContext * ctx) override;

    antlrcpp::Any visitStarExpr(SQLGrammarParser::StarExprContext * ctx) override;

    antlrcpp::Any visitIdentifierExpr(SQLGrammarParser::IdentifierExprContext * ctx) override;

    antlrcpp::Any visitGroupAggExpr(SQLGrammarParser::GroupAggExprContext * ctx) override;

    antlrcpp::Any visitParanthesesExpr(SQLGrammarParser::ParanthesesExprContext * ctx) override;

    antlrcpp::Any visitMulExpr(SQLGrammarParser::MulExprContext * ctx) override;

    antlrcpp::Any visitAddExpr(SQLGrammarParser::AddExprContext * ctx) override;

    antlrcpp::Any visitCmpExpr(SQLGrammarParser::CmpExprContext * ctx) override;

    antlrcpp::Any visitAndExpr(SQLGrammarParser::AndExprContext * ctx) override;

    antlrcpp::Any visitOrExpr(SQLGrammarParser::OrExprContext * ctx) override;

//tableReference
    antlrcpp::Any visitTableReference(SQLGrammarParser::TableReferenceContext * ctx) override;

//selectIdent
    antlrcpp::Any visitStringIdent(SQLGrammarParser::StringIdentContext * ctx) override;

//literal
    antlrcpp::Any visitLiteral(SQLGrammarParser::LiteralContext * ctx) override;
};
