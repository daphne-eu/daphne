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

#include <parser/sql/morphstore/MorphStoreSQLVisitor.h>
#include "util/BitPusher.h"

/**
 * @brief Test if the flag at the position is set and returns result
 */
//bool isBitSet(int64_t flag, int64_t position){
//    return ((flag >> position) & 1) == 1;
//}

antlrcpp::Any MorphStoreSQLVisitor::visitCmpExpr(
        SQLGrammarParser::CmpExprContext * ctx
)
{
    mlir::Location loc = utils.getLoc(ctx->start);
    std::string op = ctx->op->getText();

    antlrcpp::Any vLhs = visit(ctx->lhs);
    antlrcpp::Any vRhs = visit(ctx->rhs);

    if(!BitPusher::isBitSet(sqlFlag, (int64_t)SQLBit::codegen)){
        return nullptr;
    }

    mlir::Value lhs = utils.valueOrError(vLhs);
    mlir::Value rhs = utils.valueOrError(vRhs);

    if(op == "=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectEqOp>(
                loc, lhs, rhs
        ));
    if(op == "<>" or op == "!=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectNeqOp>(
                loc, lhs, rhs
        ));
    if(op == "<")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectLtOp>(
                loc, lhs, rhs
        ));
    if(op == "<=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectLeOp>(
                loc, lhs, rhs
        ));
    if(op == ">")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectGtOp>(
                loc, lhs, rhs
        ));
    if(op == ">=")
        return static_cast<mlir::Value>(builder.create<mlir::daphne::MorphStoreSelectGeOp>(
                loc, lhs, rhs
        ));

    throw std::runtime_error("unexpected op symbol");
}

