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


#ifndef DAPHNE_PROTOTYPE_MORPHSTORESQLVISITOR_H
#define DAPHNE_PROTOTYPE_MORPHSTORESQLVISITOR_H

#include <parser/sql/SQLVisitor.h>

class MorphStoreSQLVisitor : public SQLVisitor{
    antlrcpp::Any visitCmpExpr(SQLGrammarParser::CmpExprContext * ctx) override;
public:
    [[maybe_unused]] explicit MorphStoreSQLVisitor(mlir::OpBuilder & builder) : SQLVisitor(builder) {
    };

    MorphStoreSQLVisitor(
            mlir::OpBuilder & builder,
    std::unordered_map <std::string, mlir::Value> view_arg
    ) : SQLVisitor(builder, view_arg){};

};


#endif //DAPHNE_PROTOTYPE_MORPHSTORESQLVISITOR_H
