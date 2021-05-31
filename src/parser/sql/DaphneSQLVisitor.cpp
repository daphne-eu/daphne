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
antlrcpp::Any visitSelect(DaphneSQLGrammarParser::SelectContext * ctx){
    mlir::res;
    mlir::Value bigframe;
    try{
        //TODO JOIN
        bigframe = valueOrError(visit(ctx->fromExpr()));
    }catch(std::runtime_error &){
        throw std::runtime_error("Error during From statement. Couldn't create Frame.");
    }

    //TODO where
    //HERE WOULD BE WHERE and co.

    for(int i = 0; i < ctx->selectExpr().size; i++){
        antlrcpp::Any se_name = visit(ctx->selectExpr(i));  //returns frame name ref
        std::string scopename = "-" + i_se + "-" + se_name;

        mlir::Value se_id;
        try{
            se_id = valueOrError(symbolTable.get(scopename));
        }catch(std::runtime_error &){
            throw std::runtime_error("Error during From statement. Couldn't create Frame.");
        }

        mlir::Location loc = builder.getUnknownLoc();
        for(int v = 0; v < fj_order.size(); v++){
            if(se_name.compare(fj_order.at(v).at(0)) == 0 || (fj_order.at(v).size() > 1 && se_name.compare(fj_order.at(v).at(1)) == 0)){
                break;
            }else{
                mlir::value c_count = static_cast<mlir::Value>(
                    builder.create<mlir::daphne::NumRowOp>(
                        loc,
                        symbolTable.get(fj_order.at(v).at(0))
                    )
                );
                se_id = static_cast<mlir::Value>(builder.create<mlir::daphne::EwAddOp>(loc, se_id, c_count);
            }
        }
        //TODO correct implementation needed
        // extract column from join result
        mlir::value ex = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ExtractColumnOp>(
                loc,
                bigframe,
                se_id;
            )
        );

        //TODO correct implementation needed
        // insert extracted column into result
        res = static_cast<mlir:Value>(
            builder.create<mlir::daphne::InsertOp>(
                loc,
                res,
                ex
            )
        );
    }

    // symbolTable.put(symbolTable.popScope());
    return res;
}

antlrcpp::Any visitSubquery(DaphresneSQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->select())));
    return nullptr;
}


antlrcpp::Any visitTableIdentifierExpr(DaphneSQLGrammarParserTableIdentifierExprContext *ctx){
    std::vector<std::string> var_name = static_cast<std::vector<std::string>>>(value(ctx->var));
    try{
        mlir::Value var = valueOrError(symbolTable.get(var_name.at(0)));
        fj_order.push(var_name);
        return var;
    }catch(std::runtime_error &){
        throw std::runtime_error("Error during From statement. Couldn't create Frame.");
    }


}
//saving every operation into symboltable should make it possible for easier code reordering. and easier selection.
antlrcpp::Any visitCartesianExpr(DaphneSQLGrammarParser::CartesianExprContext * ctx)
{
    try{
        antlrcpp::Any lhs = valueOrError(visit(ctx->lhs));

        std::vector<std::string> rhs_name =  static_cast<std::vector<std::string>>>(value(ctx->rhs);
        antlrcpp::Any rhs = valueOrError(symbolTable.get(rhs_name.at(0)));
        fj_order.push(rhs_name);
        //creating join code
        mlir::Value co = static_cast<mlir::Value>(builder.create<mlir::daphne::CartesianOp>(lhs, rhs))
        return co;
    }catch{
        throw std::runtime_error("Unexpected Error during cartesian operation");
    }
}


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
antlrcpp::Any visitSelectExpr(DaphneSQLGrammarParser::SelectExprContext * ctx) {
    return visitChildren(ctx);
}


//Needs to put it's own value into the scope again (this means that it is in the current scope)
//this is a hack of the symboltable. Maybe there is a better way.
//retrurns string which needs to be looked up before use. This has todo with the implementation of from/join
antlrcpp::Any visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) {
    std::vector<std::string> names;
    std::string var = ctx->var->getText();
    names.add(var);
    try {
        antlrcpp::Any res = symbolTable.get(var);
        symbolTable.put(var, res);
        if(ctx->aka){
            symbolTable.put(ctx->aka->getText(), res);
            names.add(ctx->aka->getText());
        }
        return names;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + var + " referenced before assignment");
    }
}

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

// TODO: make to fit select
antlrcpp::Any visitIntIdent(DaphneSQLGrammarParser::IdentContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string frame = ctx->frame->getText();
    std::string id = atol(ctx->INT_POSITIVE_LITERAL()->getText().c_str());
    try {

        mlir::Value m_id = std::string scopename = "-" + (++i_se) + "-" + frame;
            static_cast<mlir::Value>(
                    builder.create<mlir::daphne::ConstantOp>(
                            loc,
                            builder.getIntegerAttr(builder.getIntegerType(64, true), id)
                    )
            );
        return frame;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + frame + " referenced before assignment");
    }
}

antlrcpp::Any visitLiteral(DaphneSQLGrammarParser::LiteralContext * ctx) {
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
