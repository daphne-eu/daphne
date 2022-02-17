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
#include <utility>

class SQLVisitor : public SQLGrammarVisitor {

    ParserUtils utils;

    mlir::OpBuilder builder;
    mlir::Value currentFrame;

//  PREFIX SHOULD NOT CONTAIN THE DOT.
    //framename, prefix
    std::unordered_map <std::string, std::string> framePrefix;
    //prefix, framename
    std::unordered_map<std::string, std::string> reverseFramePrefix;

    std::unordered_map <std::string, mlir::Value> view;
    std::unordered_map <std::string, mlir::Value> alias;

    void registerAlias(const std::string& framename, mlir::Value arg);
    std::string setFramePrefix(const std::string& framename, const std::string& prefix, bool necessary, bool ignore);

    mlir::Value fetchMLIR(const std::string& framename);    //looks name up in alias and view
    [[maybe_unused]] mlir::Value fetchAlias(const std::string& framename);   //looks name up only in alias
    std::string fetchPrefix(const std::string& framename);
    bool hasMLIR(const std::string& name);

    ScopedSymbolTable symbolTable;

public:
    [[maybe_unused]] explicit SQLVisitor(mlir::OpBuilder & builder) : utils(builder), builder(builder) {
    };

    SQLVisitor(
        mlir::OpBuilder & builder,
        std::unordered_map <std::string, mlir::Value> view_arg
    ) : utils(builder), builder(builder) {
        view = std::move(view_arg);
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
