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

    antlrcpp::Any visitSql(DaphneSQLGrammarParser::SqlContext * ctx) override;

    antlrcpp::Any visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) override;

    antlrcpp::Any visitSelect(DaphneSQLGrammarParser::SelectContext * ctx) override;

    antlrcpp::Any visitSubquery(DaphneSQLGrammarParser::SubqueryContext * ctx) override;

    antlrcpp::Any visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) override;


    antlrcpp::Any visitTableIdentifierExpr(DaphneSQLGrammarParser::TableIdentifierExprContext *ctx) override;
    //saving every operation into symboltable should make it possible for easier code reordering. and easier selection.
    antlrcpp::Any visitCartesianExpr(DaphneSQLGrammarParser::CartesianExprContext * ctx) override;


    /*
    antlrcpp::Any visitInnerJoin(DaphneSQLGrammarParser::InnerJoinContext * ctx) {
        antlrcpp::Any lhs = valueOrError(visit(ctx->lhs));
        antlrcpp::Any rhs = valueOrError(visit(ctx->rhs));
        antlrcpp::Any cond = valueOrError(visit(ctx->cond));
        //creating join code
        mlir::Value jr = static_cast<mlir::Value>(builder.create<mlir::daphne::InnerJoinOp>(lhs, rhs)) //the next to arguments must still be adressed.
        //if we put jr into the symbol table. we could add information about the frames and help find the columns.
        return jr;
    }

    antlrcpp::Any visitCrossJoin(DaphneSQLGrammarParser::CrossJoinContext * ctx) {}

    antlrcpp::Any visitNaturalJoin(DaphneSQLGrammarParser::NaturalJoinContext * ctx) {}


    antlrcpp::Any visitFullJoin(DaphneSQLGrammarParser::FullJoinContext * ctx) {}

    antlrcpp::Any visitLeftJoin(DaphneSQLGrammarParser::LeftJoinContext * ctx) {}

    antlrcpp::Any visitRightJoin(DaphneSQLGrammarParser::RightJoinContext * ctx) {}


    antlrcpp::Any visitJoinCondition(DaphneSQLGrammarParser::JoinConditionContext * ctx) {}


    antlrcpp::Any visitWhereClause(DaphneSQLGrammarParser::WhereClauseContext * ctx) {

    }


    antlrcpp::Any visitLiteralExpr(DaphneSQLGrammarParser::LiteralExprContext * ctx) {}

    antlrcpp::Any visitIdentifierExpr(DaphneSQLGrammarParser::IdentifierExprContext * ctx) {}

    antlrcpp::Any visitParanthesesExpr(DaphneSQLGrammarParser::ParanthesesExprContext * ctx) {}

    antlrcpp::Any visitMulExpr(DaphneSQLGrammarParser::MulExprContext * ctx) {}

    antlrcpp::Any visitAddExpr(DaphneSQLGrammarParser::AddExprContext * ctx) {}

    antlrcpp::Any visitCmpExpr(DaphneSQLGrammarParser::CmpExprContext * ctx) {}

    antlrcpp::Any visitLogicalExpr(DaphneSQLGrammarParser::LogicalExprContext * ctx) {}
    */
    //TODO when columns get names than this has to be updated
    antlrcpp::Any visitSelectExpr(DaphneSQLGrammarParser::SelectExprContext * ctx) override;


    //Needs to put it's own value into the scope again (this means that it is in the current scope)
    //this is a hack of the symboltable. Maybe there is a better way.
    //retrurns string which needs to be looked up before use. This has todo with the implementation of from/join
    antlrcpp::Any visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) override;

    /*
    antlrcpp::Any visitStringIdent(DaphneSQLGrammarParser::IdentContext * ctx) {

        std::string var = atol(ctx->IDENTIFIER()->getText());
        if(ctx->frame){
            try {
                return static_cast<mlir::Value>(
                    builder.create<mlir::daphne::GetOp>(
                        loc,
                        valueOrError(symbolTable.get(frame)),
                        builder.getStringAttr(var)
                    )
                );
            }
            catch(std::runtime_error &) {
                throw std::runtime_error("Frame " + frame + " referenced before assignment");
            }
        }
        try {
            return symbolTable.get(var);
        }
        catch(std::runtime_error &) {
            throw std::runtime_error("variable " + var + " referenced before assignment");
        }
    }
    */

    antlrcpp::Any visitIntIdent(DaphneSQLGrammarParser::IntIdentContext * ctx) override;

    antlrcpp::Any visitLiteral(DaphneSQLGrammarParser::LiteralContext * ctx) override;




/*

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
*/
};

#endif //SRC_PARSER_DAPHNESQL_DAPHNESQLVISITOR_H
