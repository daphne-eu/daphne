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

#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
#include "DaphneSQLGrammarParser.h"
#include "DaphneSQLGrammarVisitor.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

class DaphneSQLVisitor : public DaphneSQLGrammarVisitor {
    // By inheriting from DaphneSQLGrammarVisitor (as opposed to
    // DaphneSQLGrammarBaseVisitor), we ensure that any newly added visitor
    // function (e.g. after a change to the grammar file) needs to be
    // considered here. This is to force us not to forget anything.

    /**
     * The OpBuilder used to generate DaphneIR operations.
     */
    mlir::OpBuilder builder;
    int i_se = 0;
    std::vector<std::vector<std::string>> fj_order;

    /**
     * Maps a variable name from the input DaphneSQL script to the MLIR SSA
     * value that has been assigned to it most recently.
     */
    ScopedSymbolTable symbolTable;

public:
    DaphneSQLVisitor(mlir::OpBuilder & builder) : builder(builder) {
        //
    };

    antlrcpp::Any visitScript(DaphneSQLGrammarParser::ScriptContext * ctx) override;

    antlrcpp::Any visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) override;

    antlrcpp::Any visitSelect(DaphneSQLGrammarParser::SelectContext * ctx) override;

    antlrcpp::Any visitSubquery(DaphneSQLGrammarParser::SubqueryContext * ctx) override;

    antlrcpp::Any visitSubqueryList(DaphneSQLGrammarParser::SubqueryListContext * ctx) override;

    antlrcpp::Any visitSelectList(DaphneSQLGrammarParser::SelectListContext * ctx) override;

    antlrcpp::Any visitTableList(DaphneSQLGrammarParser::TableListContext * ctx) override;

    antlrcpp::Any visitJoinList(DaphneSQLGrammarParser::JoinListContext * ctx) override;

    antlrcpp::Any visitJoinClause(DaphneSQLGrammarParser::JoinClauseContext * ctx) override;

    antlrcpp::Any visitInnerCrossJoinClause(DaphneSQLGrammarParser::InnerCrossJoinClauseContext * ctx) override;

    antlrcpp::Any visitOuterJoinClause(DaphneSQLGrammarParser::OuterJoinClauseContext * ctx) override;

    antlrcpp::Any visitJoinCondition(DaphneSQLGrammarParser::JoinConditionContext * ctx) override;

    antlrcpp::Any visitOuterJoinType(DaphneSQLGrammarParser::OuterJoinTypeContext * ctx) override;

    antlrcpp::Any visitLiteralExpr(DaphneSQLGrammarParser::LiteralExprContext * ctx) override;

    antlrcpp::Any visitIdentifierExpr(DaphneSQLGrammarParser::IdentifierExprContext * ctx) override;

    antlrcpp::Any visitParanthesesExpr(DaphneSQLGrammarParser::ParanthesesExprContext * ctx) override;

    antlrcpp::Any visitMulExpr(DaphneSQLGrammarParser::MulExprContext * ctx) override;

    antlrcpp::Any visitAddExpr(DaphneSQLGrammarParser::AddExprContext * ctx) override;

    antlrcpp::Any visitCmpExpr(DaphneSQLGrammarParser::CmpExprContext * ctx) override;

    antlrcpp::Any visitLogicalExpr(DaphneSQLGrammarParser::LogicalExprContext * ctx) override;

    antlrcpp::Any visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) override;

    antlrcpp::Any visitIdent(DaphneSQLGrammarParser::IdentContext * ctx) override;

    antlrcpp::Any visitAlias(DaphneSQLGrammarParser::AliasContext * ctx) override;

    antlrcpp::Any visitLiteral(DaphneSQLGrammarParser::LiteralContext * ctx) override;
};

#endif //SRC_PARSER_DAPHNESQL_DAPHNESQLVISITOR_H
