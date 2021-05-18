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
#include <parser/daphnedsl/DaphneSQLVisitor.h>
#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
#include "DaphneDSLGrammarParser.h"

#include <mlir/Dialect/SCF/SCF.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>

// ****************************************************************************
// Helper functions
// ****************************************************************************

mlir::Value valueOrError(antlrcpp::Any a) {
    if(a.is<mlir::Value>())
        return a.as<mlir::Value>();
    throw std::runtime_error("something was expected to be an mlir::Value, but it was none");
}

// ****************************************************************************
// Visitor functions
// ****************************************************************************

antlrcpp::Any visitScript(DaphneSQLGrammarParser::ScriptContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) {
    return visitChildren(ctx);
}

//this needs to return a frame so that subquery is a frame.
//get a joined and cartesian product.
//on that needs to perform where clause and followed up with a projection.
//PROBLEM: due to new frame the indices are messed up. with named columns this wouldnn't be an issue
//it would be good to know the columns that we want to keep before we execute the joins.
//this would
    antlrcpp::Any tableproduct = valueOrError(visit(ctx->tableList()));
    if(where->clause)

    symbolTable.put(symbolTable.popScope());
    return res;
}

antlrcpp::Any visitSubquery(DaphresneSQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->select())));
    return nullptr;
}


antlrcpp::Any visitSelectList(DaphneSQLGrammarParser::SelectListContext * ctx) {

}

//this needs to build the cartesianproduct or needs to trackthat one must be build.
antlrcpp::Any visitTableList(DaphneSQLGrammarParser::TableListContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any visitJoinList(DaphneSQLGrammarParser::JoinListContext * ctx) {}

antlrcpp::Any visitJoinClause(DaphneSQLGrammarParser::JoinClauseContext * ctx) {}

antlrcpp::Any visitInnerCrossJoinClause(DaphneSQLGrammarParser::InnerCrossJoinClauseContext * ctx) {}

antlrcpp::Any visitOuterJoinClause(DaphneSQLGrammarParser::OuterJoinClauseContext * ctx) {}

antlrcpp::Any visitJoinCondition(DaphneSQLGrammarParser::JoinConditionContext * ctx) {}

antlrcpp::Any visitOuterJoinType(DaphneSQLGrammarParser::OuterJoinTypeContext * ctx) {}

antlrcpp::Any visitLiteralExpr(DaphneSQLGrammarParser::LiteralExprContext * ctx) {}

antlrcpp::Any visitIdentifierExpr(DaphneSQLGrammarParser::IdentifierExprContext * ctx) {}

antlrcpp::Any visitParanthesesExpr(DaphneSQLGrammarParser::ParanthesesExprContext * ctx) {}

antlrcpp::Any visitMulExpr(DaphneSQLGrammarParser::MulExprContext * ctx) {}

antlrcpp::Any visitAddExpr(DaphneSQLGrammarParser::AddExprContext * ctx) {}

antlrcpp::Any visitCmpExpr(DaphneSQLGrammarParser::CmpExprContext * ctx) {}

antlrcpp::Any visitLogicalExpr(DaphneSQLGrammarParser::LogicalExprContext * ctx) {}


//Needs to put it's own value into the scope again (this means that it is in the current scope)
//this is a hack of the symboltable. Maybe there is a better way.
antlrcpp::Any visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) {
    std::string var = ctx->var->getText();
    try {
        antlrcpp::Any res = symbolTable.get(var);
        symbolTable.put(var, res);
        if(ctx->aka){
            symbolTable.put(ctx->aka->getText(), res);
        }
        return res;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Unknown Frame " + var + " referenced before assignment");
}

antlrcpp::Any visitIdent(DaphneSQLGrammarParser::IdentContext * ctx) {

}

antlrcpp::Any visitAlias(DaphneSQLGrammarParser::AliasContext * ctx) {}

antlrcpp::Any visitLiteral(DaphneSQLGrammarParser::LiteralContext * ctx) {}
