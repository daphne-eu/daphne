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

#ifndef SRC_PARSER_SQL_SQLVISITOR_H
#define SRC_PARSER_SQL_SQLVISITOR_H

#include <parser/ParserUtils.h>
#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
#include "SQLGrammarParser.h"
#include "SQLGrammarVisitor.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <string>
#include <unordered_map>

class SQLVisitor : public SQLGrammarVisitor {

    ParserUtils utils;

    mlir::OpBuilder builder;
    mlir::Value currentFrame;
    int i_se = 0;

    //framename, prefix
    std::unordered_map <std::string, std::string> framePrefix;
    //prefix, framename
    std::unordered_map<std::string, std::string> reverseFramePrefix;

    std::unordered_map <std::string, mlir::Value> view;
    std::unordered_map <std::string, mlir::Value> alias;

    void registerAlias(std::string framename, mlir::Value arg);
    std::string setFramePrefix(std::string framename, std::string prefix, bool necessary, bool ignore);

    mlir::Value fetchMLIR(std::string framename);    //looks name up in alias and view
    mlir::Value fetchAlias(std::string framename);   //looks name up only in alias
    std::string fetchPrefix(std::string framename);
    bool hasMLIR(std::string name);

    ScopedSymbolTable symbolTable;

public:
    SQLVisitor(mlir::OpBuilder & builder) : builder(builder), utils(builder) {
    };

    SQLVisitor(
        mlir::OpBuilder & builder,
        std::unordered_map <std::string, mlir::Value> view_arg
    ) : builder(builder), utils(builder) {
        view = view_arg;
    };

    antlrcpp::Any visitScript(SQLGrammarParser::ScriptContext * ctx) override;

    antlrcpp::Any visitSql(SQLGrammarParser::SqlContext * ctx) override;

    antlrcpp::Any visitQuery(SQLGrammarParser::QueryContext * ctx) override;

    antlrcpp::Any visitSelect(SQLGrammarParser::SelectContext * ctx) override;

    antlrcpp::Any visitSubquery(SQLGrammarParser::SubqueryContext * ctx) override;

    antlrcpp::Any visitSubqueryExpr(SQLGrammarParser::SubqueryExprContext * ctx) override;

    antlrcpp::Any visitTableIdentifierExpr(SQLGrammarParser::TableIdentifierExprContext *ctx) override;

    antlrcpp::Any visitTableExpr(SQLGrammarParser::TableExprContext * ctx) override;

    antlrcpp::Any visitInnerJoin(SQLGrammarParser::InnerJoinContext * ctx) override;

    antlrcpp::Any visitCartesianExpr(SQLGrammarParser::CartesianExprContext * ctx) override;

    antlrcpp::Any visitSelectExpr(SQLGrammarParser::SelectExprContext * ctx) override;

    antlrcpp::Any visitWhereClause(SQLGrammarParser::WhereClauseContext * ctx) override;

    antlrcpp::Any visitIdentifierExpr(SQLGrammarParser::IdentifierExprContext * ctx) override;

    antlrcpp::Any visitLiteralExpr(SQLGrammarParser::LiteralExprContext * ctx) override;

    antlrcpp::Any visitParanthesesExpr(SQLGrammarParser::ParanthesesExprContext * ctx) override;

    antlrcpp::Any visitMulExpr(SQLGrammarParser::MulExprContext * ctx) override;

    antlrcpp::Any visitAddExpr(SQLGrammarParser::AddExprContext * ctx) override;

    antlrcpp::Any visitCmpExpr(SQLGrammarParser::CmpExprContext * ctx) override;

    antlrcpp::Any visitAndExpr(SQLGrammarParser::AndExprContext * ctx) override;

    antlrcpp::Any visitOrExpr(SQLGrammarParser::OrExprContext * ctx) override;

    antlrcpp::Any visitTableReference(SQLGrammarParser::TableReferenceContext * ctx) override;

    antlrcpp::Any visitStringIdent(SQLGrammarParser::StringIdentContext * ctx) override;

    antlrcpp::Any visitLiteral(SQLGrammarParser::LiteralContext * ctx) override;
};

#endif //SRC_PARSER_SQL_SQLVISITOR_H
