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

#ifndef SRC_PARSER_DAPHNESQL_DAPHNESQLVISITOR_H
#define SRC_PARSER_DAPHNESQL_DAPHNESQLVISITOR_H

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
    std::vector<std::vector<std::string>> fj_order;

    std::unordered_map <std::string, mlir::Value> view;
    std::unordered_map <std::string, mlir::Value> alias;

    void registerAlias(mlir::Value arg, std::string name);

    mlir::Value fetchMLIR(std::string name);    //looks name up in alias and view
    mlir::Value fetchAlias(std::string name);   //looks name up only in alias
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

    antlrcpp::Any visitCartesianExpr(SQLGrammarParser::CartesianExprContext * ctx) override;

    antlrcpp::Any visitSelectExpr(SQLGrammarParser::SelectExprContext * ctx) override;

    antlrcpp::Any visitTableReference(SQLGrammarParser::TableReferenceContext * ctx) override;

    antlrcpp::Any visitStringIdent(SQLGrammarParser::StringIdentContext * ctx) override;

    antlrcpp::Any visitLiteral(SQLGrammarParser::LiteralContext * ctx) override;
};

#endif //SRC_PARSER_DAPHNESQL_DAPHNESQLVISITOR_H
