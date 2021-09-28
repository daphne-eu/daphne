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
    mlir::Value res = fetch(alias, name);
    if(res != NULL){
        return res;
    }
    std::stringstream x;
    x << "Error: " << name << " does not name an Alias\n";
    throw std::runtime_error(x.str());
}

mlir::Value DaphneSQLVisitor::fetchMLIR(std::string name){
    mlir::Value res;
    res = fetch(alias, name);
    if(res != NULL){
        return res;
    }
    res = fetch(view, name);
    if(res != NULL){
        return res;
    }
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
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitSql(DaphneSQLGrammarParser::SqlContext * ctx) {
    mlir::Value res = valueOrError(visit(ctx->query()));
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitQuery(DaphneSQLGrammarParser::QueryContext * ctx) {
    mlir::Value res = valueOrError(visit(ctx->select()));
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitSelect(DaphneSQLGrammarParser::SelectContext * ctx){
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value res;

    try{
        currentFrame = valueOrError(visit(ctx->fromExpr()));
    }catch(std::runtime_error & e){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't create Frame.\n\t\t" << e.what();

        throw std::runtime_error(err_msg.str());
    }

    //TODO: WHERE Statement. This would be a good place for it.

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

            std::vector<mlir::Type> colTypes;
            for(mlir::Type t : res.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
                colTypes.push_back(t);
            for(mlir::Type t : add.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
                colTypes.push_back(t);
            mlir::Type resType = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

            mlir::Value nr_col = static_cast<mlir::Value> (builder.create<mlir::daphne::NumColsOp>(loc, utils.sizeType , add));

            res = static_cast<mlir::Value>(
                builder.create<mlir::daphne::ColBindOp>(
                    loc,
                    resType,
                    res,
                    add
                )
            );
        }catch(std::runtime_error & e){
            std::stringstream err_msg;
            err_msg << "Error during SELECT statement. Couldn't Insert Extracted Columns.\n\t\t" << e.what();
            throw std::runtime_error(err_msg.str());
        }
    }
    return res;
}

antlrcpp::Any DaphneSQLVisitor::visitSubquery(DaphneSQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneSQLVisitor::visitSubqueryExpr(DaphneSQLGrammarParser::SubqueryExprContext * ctx) {
    symbolTable.put(ctx->var->getText(), valueOrError(visit(ctx->select())));
    return nullptr;
}

antlrcpp::Any DaphneSQLVisitor::visitTableIdentifierExpr(DaphneSQLGrammarParser::TableIdentifierExprContext *ctx){
    try{
        mlir::Value var = valueOrError(visit(ctx->var));
        return var;
    }catch(std::runtime_error &){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't find Frame.";
        throw std::runtime_error(err_msg.str());
    }
}

antlrcpp::Any DaphneSQLVisitor::visitCartesianExpr(DaphneSQLGrammarParser::CartesianExprContext * ctx)
{
    try{
        mlir::Location loc = builder.getUnknownLoc();
        mlir::Value res;
        mlir::Value lhs = valueOrError(visit(ctx->lhs));
        mlir::Value rhs = valueOrError(visit(ctx->rhs));

        std::vector<mlir::Type> colTypes;
        for(mlir::Type t : lhs.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
            colTypes.push_back(t);
        for(mlir::Type t : rhs.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
            colTypes.push_back(t);
        mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

        res = static_cast<mlir::Value>(
            builder.create<mlir::daphne::CartesianOp>(
                loc,
                t,
                lhs,
                rhs
            )
        );

        return res;
    }catch(std::runtime_error &){
        throw std::runtime_error("Unexpected Error during cartesian operation");
    }
}

//*******
//* TODO:
//* needs to check if var name already in use in this scope
//* needs to correctly implement SetColLabelsPrefixOp and how to access these labels
//*******
antlrcpp::Any DaphneSQLVisitor::visitTableReference(DaphneSQLGrammarParser::TableReferenceContext * ctx) {

    mlir::Location loc = builder.getUnknownLoc();
    std::string var = ctx->var->getText();
    try {
        mlir::Value res = fetchMLIR(var);
        registerAlias(res, var);
        if(ctx->aka){
            var = ctx->aka->getText();
            registerAlias(res, var);
        }
        // builder.create<mlir::daphne::SetColLabelsPrefixOp>(loc, res, var);
        return res;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + var + " referenced before assignment");
    }
}

//****
//* Returns what selectIdent (stringIdent/intIdent) returns.
//*     This is a SSA to ExtractColumn Operation.
//* Callee: visitSelect
//* TODO: Return Rename Operation SSA based on this Operation
//****/
antlrcpp::Any DaphneSQLVisitor::visitSelectExpr(DaphneSQLGrammarParser::SelectExprContext * ctx) {
    return visitChildren(ctx);
}

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
                getSSA
            )
        );
    }catch(std::runtime_error &) {
        throw std::runtime_error("Unexpected Error. Couldn't create Extract Operation for Select");
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
