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

/*
 *  TODO:
 *      1: Relational Algebra
 *          This will make it possible for annother pass for reordering
 *          Operations and lowering them to Daphne, so that other languages
 *          Can use it.
 *      2: Named Columns
 *          Renameing column operations, such that a projection works
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
#include <sstream>

#include <cstdint>
#include <cstdlib>
#include <regex>

// ****************************************************************************
// Helper functions
// ****************************************************************************

mlir::Value valueOrError(antlrcpp::Any a) {
    if(a.is<mlir::Value>())
        return a.as<mlir::Value>();
    throw std::runtime_error("something was expected to be an mlir::Value, but it was none");
}

void DaphneSQLVisitor::registerAlias(mlir::Value arg, std::string name){
    alias[name] = arg;
}

mlir::Value fetch(std::unordered_map <std::string, mlir::Value> x, std::string name){
    auto search = x.find(name);
    if(search != x.end()){
        return search->second;
    }
    return NULL;
}

mlir::Value DaphneSQLVisitor::fetchAlias(std::string name){
    std::cout << "Fetch <" << name << "> from alias\n";

    mlir::Value res = fetch(alias, name);
    if(res != NULL){
        return res;
    }

    std::stringstream x;
    x << "Error: " << name << " was not registered with the Function registerView\n";
    throw std::runtime_error(x.str());
}

mlir::Value DaphneSQLVisitor::fetchMLIR(std::string name){
    std::cout << "FetchMLIR ";
    mlir::Value res = fetch(alias, name);
    if(res != NULL){
        // std::cout << "<" << name << "> from alias\n";
        return res;
    }
    res = fetch(view, name);
    if(res != NULL){
        // std::cout << "<" << name << "> from view\n";
        return res;
    }
    std::cout << std::endl;
    std::stringstream x;
    x << "Error: " << name << " was not registered with the Function \"registerView\" or were given an alias\n";
    throw std::runtime_error(x.str());
}

bool DaphneSQLVisitor::hasMLIR(std::string name){
    auto searchview = view.find(name);
    auto searchalias = alias.find(name);
    return (searchview != view.end() || searchalias != alias.end());
}

// ****************************************************************************
// Visitor functions
// ****************************************************************************

antlrcpp::Any DaphneSQLVisitor::visitScript(DaphneSQLGrammarParser::ScriptContext * ctx) {
    mlir::Value res = valueOrError(visitChildren(ctx));
    std::cout << "all good up until now <SCRIPT>\n";
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitSql(DaphneSQLGrammarParser::SqlContext * ctx) {
//    mlir::Value res = valueOrError(visitChildren(ctx));
    mlir::Value res = valueOrError(visit(ctx->query()));
    std::cout << "all good up until now <SQL>\n";
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) {
    mlir::Value res = valueOrError(visit(ctx->select()));
    std::cout << "all good up until now <QUERY>\n";
    return res;
}

//this needs to return a frame so that subquery is a frame.
//get a joined and cartesian product.
//on that needs to perform where clause and followed up with a projection.
//PROBLEM: due to new frame the indices are messed up. with named columns this wouldnn't be an issue
//it would be good to know the columns that we want to keep before we execute the joins.
//this would



//TODO:
//  1: Implement where
//  2: Relational Alg
//  3: Rest
antlrcpp::Any DaphneSQLVisitor::visitSelect(DaphneSQLGrammarParser::SelectContext * ctx){

    mlir::Value res;
    try{
        currentFrame = valueOrError(visit(ctx->fromExpr()));
    }catch(std::runtime_error & e){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't create Frame.\n\t\t" << e.what();

        throw std::runtime_error(err_msg.str());
    }

    //TODO: WHERE Statement. This would be a good place for it.

    //TODO> rework
    mlir::Location loc = builder.getUnknownLoc();
    res = valueOrError(visit(ctx->selectExpr(0)));
    for(auto i = 1; i < ctx->selectExpr().size(); i++){
        mlir::Value add;
        try{
            add = valueOrError(visit(ctx->selectExpr(i)));
        }catch(std::runtime_error &e){
            std::stringstream err_msg;
            err_msg << "Something went wrong in SelectExpr.\n\t\t" << e.what();
            throw std::runtime_error(err_msg.str());
        }
        try{
            mlir::Type resType = mlir::daphne::FrameType::get(
                    builder.getContext(), {utils.unknownType}
            );

            mlir::Value nr_col = static_cast<mlir::Value> (builder.create<mlir::daphne::NumColsOp>(loc, utils.sizeType , add));
            res = static_cast<mlir::Value>(
                builder.create<mlir::daphne::InsertColumnOp>(
                    loc,
                    resType,
                    res,
                    add,
                    nr_col
                )
            );
        }catch(std::runtime_error & e){
            std::stringstream err_msg;
            err_msg << "Error during SELECT statement. Couldn't Insert Extracted Columns.\n\t\t" << e.what();
            throw std::runtime_error(err_msg.str());
        }
    }
    // symbolTable.put(symbolTable.popScope());
    std::cout << "all good up until now <SELECT>\n";
    return res;
    // return nullptr;
}

antlrcpp::Any DaphneSQLVisitor::visitSubquery(DaphneSQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneSQLVisitor::visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->select())));
    return nullptr;
}

//TODO
//  1. We need to log the names of the tables somehow. OR DO WE?
//      Reason right now was that we didn't have labeled data.
//      But with labels we still need to have aliases for them so we can project
//      later. For now we don't use them.
antlrcpp::Any DaphneSQLVisitor::visitTableIdentifierExpr(DaphneSQLGrammarParser::TableIdentifierExprContext *ctx){
    // std::vector<std::string> var_name = visit(ctx->var).as<std::vector<std::string>>();
    try{
        mlir::Value var = valueOrError(visit(ctx->var));
        // mlir::Value var = fetchMLIR(var_name.at(0));
        // fj_order.push_back(var_name);
        return var;
    }catch(std::runtime_error &){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't find Frame.";
        throw std::runtime_error(err_msg.str());
    }


}

//saving every operation into symboltable should make it possible for easier code reordering. and easier selection.
//TODO:
//  1. Same as No. 1 of visitTableIdentifierExpr.
antlrcpp::Any DaphneSQLVisitor::visitCartesianExpr(DaphneSQLGrammarParser::CartesianExprContext * ctx)
{
    try{
        mlir::Value lhs = valueOrError(visit(ctx->lhs));

        // std::vector<std::string> rhs_name = visit(ctx->rhs).as<std::vector<std::string>>();
        // std::cout << "ERR1" << std::endl;
        // antlrcpp::Any rhs = valueOrError(symbolTable.get(rhs_name.at(0)));
        mlir::Value rhs = valueOrError(visit(ctx->rhs));
        // fj_order.push_back(rhs_name);

        //creating join code
        mlir::Location loc = builder.getUnknownLoc();
        mlir::Value cOp = static_cast<mlir::Value>(
            builder.create<mlir::daphne::CartesianOp>(
                loc,
                lhs.getType(),
                lhs,
                rhs)
            );
        // return nullptr;
        return cOp;
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


//Needs to put it's own value into the scope again (this means that it is in the current scope)
//this is a hack of the symboltable. Maybe there is a better way.
//retrurns string which needs to be looked up before use. This has todo with the implementation of from/join
//TODO: needs to check if var name already in use in this scope
antlrcpp::Any DaphneSQLVisitor::visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) {
    // std::vector<std::string> names;
    std::string var = ctx->var->getText();
    // names.push_back(var);
    try {
        mlir::Value res = fetchMLIR(var);
        registerAlias(res, var);
        // symbolTable.put(var, ScopedSymbolTable::SymbolInfo(res, true));
        if(ctx->aka){
            registerAlias(res, ctx->aka->getText());
            // symbolTable.put(ctx->aka->getText(), ScopedSymbolTable::SymbolInfo(res, true));
            // names.push_back(ctx->aka->getText());
        }
        return res;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + var + " referenced before assignment");
    }
}


//TODO:
//  1. Relational Alg: Not just return. Create an Operation to rename the column
//      to the alias that get set here, or the Select term.
//****
//* Returns what selectIdent (stringIdent/intIdent) returns.
//*     This is a SSA to ExtractColumn Operation.
//* Callee: visitSelect
//* TODO: Return Rename Operation SSA based on this Operation
//****
antlrcpp::Any DaphneSQLVisitor::visitSelectExpr(DaphneSQLGrammarParser::SelectExprContext * ctx) {
    return visitChildren(ctx);
}//*/

//TODO:
//  1. Do we need the frameSSA var or is just currentFrame okay?
//  2. Do we need this Function or should we move the content to SelectExpr?
//  3. Is it possible to check if the label exists?
//  4. How does the String stuff work?
//  5. Relational Alg: Change ExtractOp to something like: Projection followed by
//      Rename Operation. => this would legitimize the seperation of SelectExpr
//      and this function.
//  6. For the rename we need to return getSTR too so the rename always renames the rows.

//****
//* Returns
//*     A SSA to ExtractColumn Operation.
//* Callee: visitSelectExpr
//****
antlrcpp::Any DaphneSQLVisitor::visitStringIdent(DaphneSQLGrammarParser::StringIdentContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string getSTR;
    std::string columnSTR = ctx->var->getText();
    std::string frameSTR = "";

    mlir::Value frameSSA = currentFrame;     //TODO: define currentFrame;
    mlir::Value getSSA;

    if(ctx->frame){
        if(!hasMLIR(ctx->frame->getText())){ //we can do this, because the frame we reference musst be known.
            throw std::runtime_error("Unknown Frame: " + ctx->frame->getText() + " use before declaration during selection");
        }
        frameSTR = ctx->frame->getText() + ".";
    }

    getSTR = frameSTR+columnSTR;
    // Remove quotation marks.
    // getSTR = getSTR.substr(1, getSTR.size() - 2); //not needed

    // Replace escape sequences.
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\b)"), "\b");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\f)"), "\f");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\n)"), "\n");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\r)"), "\r");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\t)"), "\t");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\\")"), "\"");
    getSTR = std::regex_replace(getSTR, std::regex(R"(\\\\)"), "\\");
    getSSA = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, getSTR)
    );

    mlir::Type resType = mlir::daphne::FrameType::get(
            builder.getContext(), {utils.unknownType}
    );

    try{
        return static_cast<mlir::Value>(
            builder.create<mlir::daphne::ExtractColOp>(
                loc,
                resType,
                frameSSA,
                getSSA    //Probs doesn't work
            )
        );
    }catch(std::runtime_error &) {
        throw std::runtime_error("Unexpected Error. Couldn't create Extract Operation for Select");
    }
}//*/
/*
// TODO: make to fit select
// TODO: Better Throw
// TODO: MAKE TO FIT unordered/map URGENT!
//COMPLETE REWORK or DISCARD
antlrcpp::Any DaphneSQLVisitor::visitIntIdent(DaphneSQLGrammarParser::IntIdentContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string frame = ctx->frame->getText();
    int64_t id = atol(ctx->INT_POSITIVE_LITERAL()->getText().c_str());

    if(!hasMLIR(frame)){
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
// */
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
