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
#include <parser/daphnesql/DaphneSQLVisitor.h>
#include <parser/ScopedSymbolTable.h>

#include "antlr4-runtime.h"
//#include "DaphneDSLGrammarParser.h"

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

antlrcpp::Any DaphneSQLVisitor::visitScript(DaphneSQLGrammarParser::ScriptContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneSQLVisitor::visitSql(DaphneSQLGrammarParser::SqlContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneSQLVisitor::visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) {
    return visitChildren(ctx);
}

//this needs to return a frame so that subquery is a frame.
//get a joined and cartesian product.
//on that needs to perform where clause and followed up with a projection.
//PROBLEM: due to new frame the indices are messed up. with named columns this wouldnn't be an issue
//it would be good to know the columns that we want to keep before we execute the joins.
//this would
antlrcpp::Any DaphneSQLVisitor::visitSelect(DaphneSQLGrammarParser::SelectContext * ctx){
    /*
    mlir::Value res;
    mlir::Value bigframe;
    try{
        //TODO JOIN
        bigframe = valueOrError(visit(ctx->fromExpr()));
    }catch(std::runtime_error &){
        throw std::runtime_error("Error during From statement. Couldn't create Frame.");
    }

    //TODO where
    //HERE WOULD BE WHERE and co.

    for(int i = 0; i < ctx->selectExpr().size(); i++){
        antlrcpp::Any se_name = visit(ctx->selectExpr(i));  //returns frame name ref
        std::string se_name_str = se_name.as<std::string>();
        std::string scopename = "-" + i_se;
        scopename.append("-");
        scopename.append(se_name_str);

        mlir::Value se_id;
        try{
            se_id = valueOrError(symbolTable.get(scopename));
        }catch(std::runtime_error &){
            throw std::runtime_error("Error during From statement. Couldn't create Frame.");
        }

        mlir::Location loc = builder.getUnknownLoc();
        for(int v = 0; v < fj_order.size(); v++){
            if(se_name_str.compare(fj_order.at(v).at(0)) == 0 || (fj_order.at(v).size() > 1 && se_name_str.compare(fj_order.at(v).at(1)) == 0)){
                break;
            }else{
                mlir::Value c_count = static_cast<mlir::Value>(
                    builder.create<mlir::daphne::NumColOp>(
                        loc,
                        valueOrError(symbolTable.get(fj_order.at(v).at(0)))
                    )
                );
                se_id = static_cast<mlir::Value>(builder.create<mlir::daphne::EwAddOp>(loc, se_id, c_count));
            }
        }
        //TODO correct implementation needed
        // extract column from join result
        mlir::Value ex = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ExtractColumnOp>(
                loc,
                bigframe,
                se_id
            )
        );

        //TODO correct implementation needed
        // insert extracted column into result
        res = static_cast<mlir::Value>(
            builder.create<mlir::daphne::InsertColumnOp>(
                loc,
                res,
                bigframe,
                ex
            )
        );
    }

    // symbolTable.put(symbolTable.popScope());
    return res;
    */
}

antlrcpp::Any DaphneSQLVisitor::visitSubquery(DaphneSQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneSQLVisitor::visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->select())));
    return nullptr;
}


antlrcpp::Any DaphneSQLVisitor::visitTableIdentifierExpr(DaphneSQLGrammarParser::TableIdentifierExprContext *ctx){
    std::vector<std::string> var_name = visit(ctx->var).as<std::vector<std::string>>();
    try{
        mlir::Value var = valueOrError(symbolTable.get(var_name.at(0)));
        fj_order.push_back(var_name);
        return var;
    }catch(std::runtime_error &){
        throw std::runtime_error("Error during From statement. Couldn't create Frame.");
    }


}
//saving every operation into symboltable should make it possible for easier code reordering. and easier selection.
antlrcpp::Any DaphneSQLVisitor::visitCartesianExpr(DaphneSQLGrammarParser::CartesianExprContext * ctx)
{
    try{
        antlrcpp::Any lhs = valueOrError(visit(ctx->lhs));

        std::vector<std::string> rhs_name = visit(ctx->rhs).as<std::vector<std::string>>();
        antlrcpp::Any rhs = valueOrError(symbolTable.get(rhs_name.at(0)));
        fj_order.push_back(rhs_name);
        //creating join code
        // mlir::Value co = static_cast<mlir::Value>(builder.create<mlir::daphne::CartesianOp>(lhs, rhs));
        return nullptr;// return co;
    }catch(std::runtime_error &){
        throw std::runtime_error("Unexpected Error during cartesian operation");
    }
}


/*
antlrcpp::Any DaphneSQLVisitor::visitInnerJoin(DaphneSQLGrammarParser::InnerJoinContext * ctx) {
    antlrcpp::Any lhs = valueOrError(visit(ctx->lhs));
    antlrcpp::Any rhs = valueOrError(visit(ctx->rhs));
    antlrcpp::Any cond = valueOrError(visit(ctx->cond));
    //creating join code
    mlir::Value jr = static_cast<mlir::Value>(builder.create<mlir::daphne::InnerJoinOp>(lhs, rhs)) //the next to arguments must still be adressed.
    //if we put jr into the symbol table. we could add information about the frames and help find the columns.
    return jr;
}

antlrcpp::Any DaphneSQLVisitor::visitCrossJoin(DaphneSQLGrammarParser::CrossJoinContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitNaturalJoin(DaphneSQLGrammarParser::NaturalJoinContext * ctx) {}


antlrcpp::Any DaphneSQLVisitor::visitFullJoin(DaphneSQLGrammarParser::FullJoinContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitLeftJoin(DaphneSQLGrammarParser::LeftJoinContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitRightJoin(DaphneSQLGrammarParser::RightJoinContext * ctx) {}


antlrcpp::Any DaphneSQLVisitor::visitJoinCondition(DaphneSQLGrammarParser::JoinConditionContext * ctx) {}


antlrcpp::Any DaphneSQLVisitor::visitWhereClause(DaphneSQLGrammarParser::WhereClauseContext * ctx) {

}


antlrcpp::Any DaphneSQLVisitor::visitLiteralExpr(DaphneSQLGrammarParser::LiteralExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitIdentifierExpr(DaphneSQLGrammarParser::IdentifierExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitParanthesesExpr(DaphneSQLGrammarParser::ParanthesesExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitMulExpr(DaphneSQLGrammarParser::MulExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitAddExpr(DaphneSQLGrammarParser::AddExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitCmpExpr(DaphneSQLGrammarParser::CmpExprContext * ctx) {}

antlrcpp::Any DaphneSQLVisitor::visitLogicalExpr(DaphneSQLGrammarParser::LogicalExprContext * ctx) {}
*/
//TODO when columns get names than this has to be updated
antlrcpp::Any DaphneSQLVisitor::visitSelectExpr(DaphneSQLGrammarParser::SelectExprContext * ctx) {
    return visitChildren(ctx);
}


//Needs to put it's own value into the scope again (this means that it is in the current scope)
//this is a hack of the symboltable. Maybe there is a better way.
//retrurns string which needs to be looked up before use. This has todo with the implementation of from/join
//TODO: needs to check if var name already in use in this scope
antlrcpp::Any DaphneSQLVisitor::visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) {
    std::vector<std::string> names;
    std::string var = ctx->var->getText();
    names.push_back(var);
    try {
        antlrcpp::Any res = symbolTable.get(var);
        symbolTable.put(var, ScopedSymbolTable::SymbolInfo(res, true));
        if(ctx->aka){
            symbolTable.put(ctx->aka->getText(), ScopedSymbolTable::SymbolInfo(res, true));
            names.push_back(ctx->aka->getText());
        }
        return names;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + var + " referenced before assignment");
    }
}

/*
antlrcpp::Any DaphneSQLVisitor::visitStringIdent(DaphneSQLGrammarParser::IdentContext * ctx) {

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

// TODO: make to fit select
// TODO: Better Throw
antlrcpp::Any DaphneSQLVisitor::visitIntIdent(DaphneSQLGrammarParser::IntIdentContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string frame = ctx->frame->getText();
    int64_t id = atol(ctx->INT_POSITIVE_LITERAL()->getText().c_str());

    if(!symbolTable.has(frame)){
        throw std::runtime_error("Unknown Frame: " + frame + " use before declaration");
    }
    try {

        std::string scopename = "-" + (++i_se);
        scopename.append("-");
        scopename.append(frame);

        mlir::Value m_id =
            static_cast<mlir::Value>(
                    builder.create<mlir::daphne::ConstantOp>(
                            loc,
                            builder.getIntegerAttr(builder.getIntegerType(64, true), id)
                    )
            );
        symbolTable.put(scopename, ScopedSymbolTable::SymbolInfo(m_id, true));
        return frame;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Problem with given ID for Select Identifier");
    }
}

antlrcpp::Any DaphneSQLVisitor::visitLiteral(DaphneSQLGrammarParser::LiteralContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
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
        double val = atof(lit->getText().c_str());
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                        loc,
                        builder.getF64FloatAttr(val)
                )
        );
    }
    throw std::runtime_error("unexpected literal");
}
