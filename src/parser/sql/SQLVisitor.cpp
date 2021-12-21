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
#include <parser/sql/SQLVisitor.h>

#include "antlr4-runtime.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#include <cstdint>
#include <cstdlib>
#include <regex>

#include <iostream>

// ****************************************************************************
// Helper functions
// ****************************************************************************

template<typename T>
T fetch(std::unordered_map <std::string, T> x, std::string name){
    auto search = x.find(name);
    if(search != x.end()){
        return search->second;
    }
    return NULL;
}

template<>
std::string fetch<std::string>(std::unordered_map <std::string, std::string> x, std::string name){
    auto search = x.find(name);
    if(search != x.end()){
        return search->second;
    }
    return "";
}

void SQLVisitor::registerAlias(std::string name, mlir::Value arg){
    alias[name] = arg;
}

std::string SQLVisitor::setFramePrefix(std::string framename, std::string prefix, bool necessary = true, bool ignore = false){
    bool frameHasPrefix = !fetch<std::string>(framePrefix, framename).empty();
    if(frameHasPrefix){
        if(necessary){
            std::stringstream x;
            x << "Error: " << framename << " is marked as necessary for Prefix generation, but already got a Prefix. Please consider an Alias\n";
            throw std::runtime_error(x.str());
        }else{
            return "";
        }
    }

    bool inuse = !fetch<std::string>(reverseFramePrefix, prefix).empty() && !ignore;
    std::string newPrefix = prefix;
    int i = 1;
    while(inuse){
        std::stringstream x;
        x << prefix << i;
        newPrefix = x.str();
        inuse = !fetch<std::string>(reverseFramePrefix, newPrefix).empty();
    }

    if(!ignore){
        reverseFramePrefix[newPrefix] = framename;
    }
    framePrefix[framename] = newPrefix;

    return newPrefix;
}

mlir::Value SQLVisitor::fetchAlias(std::string name){
    mlir::Value res = fetch<mlir::Value>(alias, name);
    if(res != NULL){
        return res;
    }
    std::stringstream x;
    x << "Error: " << name << " does not name an Alias\n";
    throw std::runtime_error(x.str());
}

mlir::Value SQLVisitor::fetchMLIR(std::string name){
    mlir::Value res;
    res = fetch<mlir::Value>(alias, name);
    if(res != NULL){
        return res;
    }
    res = fetch<mlir::Value>(view, name);
    if(res != NULL){
        return res;
    }
    std::stringstream x;
    x << "Error: " << name << " was not registered with the Function \"registerView\" or were given an alias\n";
    throw std::runtime_error(x.str());
}

std::string SQLVisitor::fetchPrefix(std::string name){
    std::string prefix = fetch<std::string>(framePrefix, name);
    if(!prefix.empty()){
        return prefix;
    }
    return "";
}

bool SQLVisitor::hasMLIR(std::string name){
    auto searchview = view.find(name);
    auto searchalias = alias.find(name);
    return (searchview != view.end() || searchalias != alias.end());
}

// mlir::Value SQLVisitor::getEnum(const std::string & func){
//     mlir::Location loc = builder.getUnknownLoc();
//     if(func == "count"){
//         // return builder.create<mlir::daphne::ConstantOp>(
//         //         loc,
//         //         builder.getTypeArrayAttr()
//         // )
//     }
//     if(func == "sum"){
//
//     }
//     if(func == "min"){
//
//     }
//     if(func == "max"){
//
//     }
//     if(func == "avg"){
//
//     }
//     return nullptr;
// }

bool isBitSet(int64_t flag, int64_t position){
    return ((flag >> position) & 1) == 1;
}

void setBit(int64_t& flag, int64_t position, int64_t val){
    val = !!val;
    flag ^= (-val ^ flag) & (0b1 << position);
}

void toggleBit(int64_t& flag, int64_t position){
    setBit(flag, position, !isBitSet(flag, position));
}

mlir::Value createStringConstant(mlir::OpBuilder builder, std::string str){
    mlir::Location loc = builder.getUnknownLoc();
    // Replace escape sequences.
    str = std::regex_replace(str, std::regex(R"(\\b)"), "\b");
    str = std::regex_replace(str, std::regex(R"(\\f)"), "\f");
    str = std::regex_replace(str, std::regex(R"(\\n)"), "\n");
    str = std::regex_replace(str, std::regex(R"(\\r)"), "\r");
    str = std::regex_replace(str, std::regex(R"(\\t)"), "\t");
    str = std::regex_replace(str, std::regex(R"(\\\")"), "\"");
    str = std::regex_replace(str, std::regex(R"(\\\\)"), "\\");
    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::ConstantOp>(loc, str)
    );
}
// ****************************************************************************
// Visitor functions
// ****************************************************************************

//script
antlrcpp::Any SQLVisitor::visitScript(SQLGrammarParser::ScriptContext * ctx) {
    mlir::Value res = utils.valueOrError(visitChildren(ctx));
    return res;
}

//sql
antlrcpp::Any SQLVisitor::visitSql(SQLGrammarParser::SqlContext * ctx) {
    mlir::Value res = utils.valueOrError(visit(ctx->query()));
    mlir::Location loc = builder.getUnknownLoc();
    builder.create<mlir::daphne::ConstantOp>(loc, GroupEnum::SUM);
    return res;
}

//query
antlrcpp::Any SQLVisitor::visitQuery(SQLGrammarParser::QueryContext * ctx) {
    mlir::Value res = utils.valueOrError(visit(ctx->select()));
    return res;
}

//select
antlrcpp::Any SQLVisitor::visitSelect(SQLGrammarParser::SelectContext * ctx){
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value res;
    setBit(sqlFlag, (int64_t)SQLBit::codegen, 1); //activates code generation in generalExpr.

    try{
        currentFrame = utils.valueOrError(visit(ctx->tableExpr()));
    }catch(std::runtime_error & e){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't create Frame.\n\t\t" << e.what();

        throw std::runtime_error(err_msg.str());
    }

    if(ctx->whereClause()){
        currentFrame = utils.valueOrError(visit(ctx->whereClause()));
    }

    if(ctx->groupByClause()){
        setBit(sqlFlag, (int64_t)SQLBit::group, 1);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 0);
        visit(ctx->groupByClause());
        for(auto i = 0; i < ctx->selectExpr().size(); i++){
            visit(ctx->selectExpr(i));
        }
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);
        currentFrame = visit(ctx->groupByClause());
    }

    res = utils.valueOrError(visit(ctx->selectExpr(0)));
    for(auto i = 1; i < ctx->selectExpr().size(); i++){
        mlir::Value add;
        try{
            add = utils.valueOrError(visit(ctx->selectExpr(i)));
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

//subquery
antlrcpp::Any SQLVisitor::visitSubquery(SQLGrammarParser::SubqueryContext * ctx) {
    return visitChildren(ctx);
}

//subqueryExpr
antlrcpp::Any SQLVisitor::visitSubqueryExpr(SQLGrammarParser::SubqueryExprContext * ctx) {
    return nullptr;
}

//SelectExpr
//****
//* Returns what selectIdent (stringIdent/intIdent) returns.
//*     This is a SSA to ExtractColumn Operation.
//* Callee: visitSelect
//* TODO: Return Rename Operation SSA based on this Operation
//****/
antlrcpp::Any SQLVisitor::visitSelectExpr(SQLGrammarParser::SelectExprContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Value matrix;
    mlir::Value expr = utils.valueOrError(visit(ctx->var));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return expr;
    }

    if(expr.getType().isa<mlir::daphne::MatrixType>()){
        matrix = expr;
    }else{
        mlir::Value numRow = static_cast<mlir::Value>(
            builder.create<mlir::daphne::NumRowsOp>(
                loc,
                utils.sizeType,
                currentFrame
            ));

        mlir::Value one = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(
                loc,
                builder.getIntegerAttr(builder.getIntegerType(64, true), 1)
            ));

        matrix = static_cast<mlir::Value>(
            builder.create<mlir::daphne::FillOp>(
                loc,
                utils.matrixOf(expr),
                expr,
                numRow,
                utils.castSizeIf(one)
            ));
    }

    //make a Frame from the Matrix.
    std::vector<mlir::Type> colTypes;
    std::vector<mlir::Value> cols;
    std::vector<mlir::Value> labels;

    colTypes.push_back(matrix.getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());
    cols.push_back(matrix);

    std::string label;
    if(ctx->aka){
        label = ctx->aka->getText();
    }else{
        label = ctx->getText();
    }
    labels.push_back(static_cast<mlir::Value>(
            builder.create<mlir::daphne::ConstantOp>(loc, label)
    ));

    mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
    mlir::Value result = static_cast<mlir::Value>(
        builder.create<mlir::daphne::CreateFrameOp>(loc, t, cols, labels)
    );
    return result;
}

//tableExpr TODO    mlir::Location loc = builder.getUnknownLoc();

antlrcpp::Any SQLVisitor::visitTableExpr(SQLGrammarParser::TableExprContext * ctx) {
    currentFrame = utils.valueOrError(visit(ctx->fromExpr()));
    for(auto i = 0; i < ctx->joinExpr().size(); i++){
        currentFrame = utils.valueOrError(visit(ctx->joinExpr(i)));
    }
    return currentFrame;
}

//fromExpr
antlrcpp::Any SQLVisitor::visitTableIdentifierExpr(SQLGrammarParser::TableIdentifierExprContext *ctx){
    try{
        mlir::Value var = utils.valueOrError(visit(ctx->var));
        return var;
    }catch(std::runtime_error &){
        std::stringstream err_msg;
        err_msg << "Error during From statement. Couldn't find Frame.";
        throw std::runtime_error(err_msg.str());
    }
}

antlrcpp::Any SQLVisitor::visitCartesianExpr(SQLGrammarParser::CartesianExprContext * ctx) {
    try{
        mlir::Location loc = builder.getUnknownLoc();
        mlir::Value res;
        mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
        mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

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

//joinExpr TODO
antlrcpp::Any SQLVisitor::visitInnerJoin(SQLGrammarParser::InnerJoinContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Value tojoin = utils.valueOrError(visit(ctx->var));
    //rhs is join
    //lhs is currentFrame
    mlir::Value rhsName = utils.valueOrError(visit(ctx->rhs));
    mlir::Value lhsName = utils.valueOrError(visit(ctx->lhs));

    std::vector<mlir::Type> colTypes;
    for(mlir::Type t : currentFrame.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    for(mlir::Type t : tojoin.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
        colTypes.push_back(t);
    mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::InnerJoinOp>(
            loc,
            t,
            currentFrame,
            tojoin,
            rhsName,
            lhsName
        ));
}

//whereClause TODO
antlrcpp::Any SQLVisitor::visitWhereClause(SQLGrammarParser::WhereClauseContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    //WE Create a Matrix of the same length as the frame rows got.
    // Filled with zeros called a.
    // we compare a with the result of GeneralExpr called b as follows:
    // a != b
    // if b is a scalar than all equals true.
    // if b is a vector than elementwise.
    mlir::Value filter;
    mlir::Value expr = utils.valueOrError(visit(ctx->cond));
    if(expr.getType().isa<mlir::daphne::MatrixType>()){
        filter = expr;
    }else{
        // mlir::Value numRow = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::NumRowsOp>(
        //         loc,
        //         utils.sizeType,
        //         currentFrame
        //     ));
        //
        // mlir::Value one = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::ConstantOp>(
        //         loc,
        //         builder.getIntegerAttr(builder.getIntegerType(64, true), 1)
        //     ));
        //
        // filter = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::FillOp>(
        //         loc,
        //         utils.matrixOf(expr),
        //         expr,
        //         numRow,
        //         utils.castSizeIf(one)
        //     ));
    }

    mlir::Value v = static_cast<mlir::Value>(
        builder.create<mlir::daphne::FilterRowOp>(
            loc,
            currentFrame.getType(),
            currentFrame,
            filter
        )
    );
    return v;
}

//groupByClause
antlrcpp::Any SQLVisitor::visitGroupByClause(SQLGrammarParser::GroupByClauseContext * ctx) {
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        for(auto i = 0; i < ctx->selectIdent().size(); i++){
            groupName[ctx->selectIdent(i)->getText()] = 1;
        }
        if(ctx->havingClause()){
            visit(ctx->havingClause());
        }
    }else{//TODO: all of it.
        //Create Grouping Operation. (Visist select Idents)
        for(int i = 0; i < columnName.size(); i++){
            std::cout << columnName[i] << "\t" << functionName[i] << "\n";
        }
    }
    return currentFrame;
}

//havingClause TODO same as where
antlrcpp::Any SQLVisitor::visitHavingClause(SQLGrammarParser::HavingClauseContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    //WE Create a Matrix of the same length as the frame rows got.
    // Filled with zeros called a.
    // we compare a with the result of GeneralExpr called b as follows:
    // a != b
    // if b is a scalar than all equals true.
    // if b is a vector than elementwise.
    mlir::Value filter;
    mlir::Value expr = utils.valueOrError(visit(ctx->cond));
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return expr;
    }
    if(expr.getType().isa<mlir::daphne::MatrixType>()){
        filter = expr;
    }else{
        // mlir::Value numRow = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::NumRowsOp>(
        //         loc,
        //         utils.sizeType,
        //         currentFrame
        //     ));
        //
        // mlir::Value one = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::ConstantOp>(
        //         loc,
        //         builder.getIntegerAttr(builder.getIntegerType(64, true), 1)
        //     ));
        //
        // filter = static_cast<mlir::Value>(
        //     builder.create<mlir::daphne::FillOp>(
        //         loc,
        //         utils.matrixOf(expr),
        //         expr,
        //         numRow,
        //         utils.castSizeIf(one)
        //     ));
    }

    mlir::Value v = static_cast<mlir::Value>(
        builder.create<mlir::daphne::FilterRowOp>(
            loc,
            currentFrame.getType(),
            currentFrame,
            filter
        )
    );
    return v;
}

//generalExpr
antlrcpp::Any SQLVisitor::visitLiteralExpr(SQLGrammarParser::LiteralExprContext * ctx) {
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return currentFrame;
    }else{
        return utils.valueOrError(visit(ctx->literal()));
    }
}

antlrcpp::Any SQLVisitor::visitIdentifierExpr(SQLGrammarParser::IdentifierExprContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Type vt = utils.unknownType;

    if(isBitSet(sqlFlag, (int64_t)SQLBit::group)){ //TODO
        if(!isBitSet(sqlFlag, (int64_t)SQLBit::agg)){
            if(groupName[ctx->selectIdent()->getText()] == 0){
                std::stringstream err_msg;
                err_msg << "Error during a generalExpr. \"" << ctx->selectIdent()->getText()
                    << "\" Must be part of the Group Expression or have an Aggregation Function";
                throw std::runtime_error(err_msg.str());
            }
        }
    }

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return currentFrame;
    }
    mlir::Value colname = utils.valueOrError(visit(ctx->selectIdent()));
    mlir::Type resTypeCol = mlir::daphne::FrameType::get(
            builder.getContext(), {vt}
    );
    mlir::Value col = static_cast<mlir::Value>(
        builder.create<mlir::daphne::ExtractColOp>(
            loc,
            resTypeCol,
            currentFrame,
            colname
        )
    );

    mlir::Type resType = utils.matrixOf(vt);
    return static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(
            loc,
            resType,
            col
    ));
}

antlrcpp::Any SQLVisitor::visitGroupAggExpr(SQLGrammarParser::GroupAggExprContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Value res_val;
    std::string newColumnName = "group_" + ctx->var->getText();

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::group)){  //Not allowed Function Call
        throw std::runtime_error("Use of an aggregation function with out a group clause");
    }
    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){ //create Column pre Group
        if(isBitSet(sqlFlag, (int64_t)SQLBit::agg)){
            throw std::runtime_error("Nested Aggregation Functions");
        }

        columnName.push_back(newColumnName); //TODO: Pushback a MLIR::Object that has a string for this row.
        functionName.push_back(ctx->func->getText()); //TODO: Pushback a function Enum?!

        setBit(sqlFlag, (int64_t)SQLBit::agg, 1);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 1);
        mlir::Value expr = utils.valueOrError(visit(ctx->generalExpr()));
        setBit(sqlFlag, (int64_t)SQLBit::agg, 0);
        setBit(sqlFlag, (int64_t)SQLBit::codegen, 0);

        mlir::Value matrix;
        //make expr to a matrix
        if(expr.getType().isa<mlir::daphne::MatrixType>()){
            matrix = expr;
        }else{
            mlir::Value numRow = static_cast<mlir::Value>(
                builder.create<mlir::daphne::NumRowsOp>(
                    loc,
                    utils.sizeType,
                    currentFrame
                ));

            mlir::Value one = static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                    loc,
                    builder.getIntegerAttr(builder.getIntegerType(64, true), 1)
                ));

            matrix = static_cast<mlir::Value>(
                builder.create<mlir::daphne::FillOp>(
                    loc,
                    utils.matrixOf(expr),
                    expr,
                    numRow,
                    utils.castSizeIf(one)
                ));
        }

        //make a Frame from the Matrix.
        std::vector<mlir::Type> colTypes;
        std::vector<mlir::Value> cols;
        std::vector<mlir::Value> labels;

        colTypes.push_back(matrix.getType().dyn_cast<mlir::daphne::MatrixType>().getElementType());
        cols.push_back(matrix);
        labels.push_back(createStringConstant(builder, newColumnName));

        mlir::Type t = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        mlir::Value add = static_cast<mlir::Value>(
            builder.create<mlir::daphne::CreateFrameOp>(loc, t, cols, labels)
        );

        //ADD Column to currentFrame
        std::vector<mlir::Type> currentFrame_colTypes;
        for(mlir::Type t : currentFrame.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
            currentFrame_colTypes.push_back(t);
        for(mlir::Type t : add.getType().dyn_cast<mlir::daphne::FrameType>().getColumnTypes())
            currentFrame_colTypes.push_back(t);
        mlir::Type resType = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

        currentFrame = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ColBindOp>(
                loc,
                resType,
                currentFrame,
                add
            )
        );
        res_val = currentFrame;



    }else{ //Get Column after Group

        //Retriev Column from grouped currentFrame
        mlir::Type vt = utils.unknownType;

        mlir::Value rowname = utils.valueOrError(createStringConstant(builder, newColumnName + ctx->func->getText()));
        mlir::Type resTypeRow = mlir::daphne::FrameType::get(
                builder.getContext(), {vt}
        );
        mlir::Value row = static_cast<mlir::Value>(
            builder.create<mlir::daphne::ExtractColOp>(
                loc,
                resTypeRow,
                currentFrame,
                rowname
            )
        );

        mlir::Type resType = utils.matrixOf(vt);
        res_val = static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(
                loc,
                resType,
                row
        ));

    }
    return res_val;
}

antlrcpp::Any SQLVisitor::visitParanthesesExpr(SQLGrammarParser::ParanthesesExprContext * ctx) {
    return utils.valueOrError(visit(ctx->generalExpr()));
}

antlrcpp::Any SQLVisitor::visitMulExpr(SQLGrammarParser::MulExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return rhs;
    }

    if(op == "*")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwMulOp>(loc, lhs, rhs));
    if(op == "/")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwDivOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitAddExpr(SQLGrammarParser::AddExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return rhs;
    }

    if(op == "+")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAddOp>(loc, lhs, rhs));
    if(op == "-")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwSubOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitCmpExpr(SQLGrammarParser::CmpExprContext * ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return rhs;
    }

    if(op == "=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwEqOp>(loc, lhs, rhs));
    if(op == "<>")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwNeqOp>(loc, lhs, rhs));
    if(op == "<")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwLtOp>(loc, lhs, rhs));
    if(op == "<=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwLeOp>(loc, lhs, rhs));
    if(op == ">")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwGtOp>(loc, lhs, rhs));
    if(op == ">=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::EwGeOp>(loc, lhs, rhs));

    throw std::runtime_error("unexpected op symbol");
}

antlrcpp::Any SQLVisitor::visitAndExpr(SQLGrammarParser::AndExprContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return rhs;
    }

    return static_cast<mlir::Value>(builder.create<mlir::daphne::EwAndOp>(loc, lhs, rhs));
}

antlrcpp::Any SQLVisitor::visitOrExpr(SQLGrammarParser::OrExprContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value lhs = utils.valueOrError(visit(ctx->lhs));
    mlir::Value rhs = utils.valueOrError(visit(ctx->rhs));

    if(!isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return rhs;
    }

    return static_cast<mlir::Value>(builder.create<mlir::daphne::EwOrOp>(loc, lhs, rhs));
}

//tableReference
//*******
//* TODO:
//* needs to check if var name already in use in this scope
//* needs to correctly implement SetColLabelsPrefixOp and how to access these labels
//*******
antlrcpp::Any SQLVisitor::visitTableReference(SQLGrammarParser::TableReferenceContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string var = ctx->var->getText();
    std::string prefix = ctx->var->getText();
    try {
        mlir::Value res = fetchMLIR(var);
        registerAlias(var, res);
        if(ctx->aka){
            std::string aka = ctx->aka->getText();
            registerAlias(aka, res);
            prefix = setFramePrefix(aka, aka);
        }

        setFramePrefix(var, prefix, !ctx->aka, ctx->aka);

        mlir::Value prefixSSA = static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(loc, prefix)
        );

        res = static_cast<mlir::Value>(
            builder.create<mlir::daphne::SetColLabelsPrefixOp>(
                loc,
                res.getType().dyn_cast<mlir::daphne::FrameType>().withSameColumnTypes(),
                res,
                prefixSSA
            )
        );
        return res;
    }
    catch(std::runtime_error &) {
        throw std::runtime_error("Frame " + var + " referenced before assignment");
    }
}

//selectIdent //rowReference
//****
//* Returns
//*     A SSA to ExtractColumn Operation.
//* Callee: visitSelectExpr
//****
antlrcpp::Any SQLVisitor::visitStringIdent(SQLGrammarParser::StringIdentContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();

    std::string getSTR;
    std::string columnSTR = ctx->var->getText();
    std::string frameSTR = "";

    if(ctx->frame){
        if(!hasMLIR(ctx->frame->getText())){
            throw std::runtime_error("Unknown Frame: " + ctx->frame->getText() + " use before declaration during selection");
        }
        std::string framePrefix = fetchPrefix(ctx->frame->getText());
        if(!framePrefix.empty()){
            frameSTR = framePrefix + ".";
        }
    }
    getSTR = frameSTR+columnSTR;
    return createStringConstant(builder, getSTR);
}

//literal
antlrcpp::Any SQLVisitor::visitLiteral(SQLGrammarParser::LiteralContext * ctx) {
    mlir::Location loc = builder.getUnknownLoc();
    if(auto lit = ctx->INT_LITERAL()) {
        int64_t val = atol(lit->getText().c_str());
        return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(
                        loc, val
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
