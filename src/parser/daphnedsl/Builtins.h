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

#ifndef SRC_PARSER_DAPHNEDSL_BUILTINS_H
#define SRC_PARSER_DAPHNEDSL_BUILTINS_H

#include "ir/daphneir/Daphne.h"

#include "antlr4-runtime.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

#include <vector>
#include <string>

template<typename T>
struct Builtin
{
    std::vector<unsigned int> expectedNumOfParams;

    Builtin(std::vector<unsigned int> expectedNumOfParams);
    virtual ~Builtin();

    mlir::LogicalResult checkNumParams(mlir::Location &loc, llvm::StringRef name, size_t size);
    virtual T create(mlir::OpBuilder builder,
                     mlir::Location &loc,
                     mlir::ValueRange values) = 0;
};

struct PrintBuiltin : public Builtin<mlir::daphne::PrintOp>
{
    using Builtin<mlir::daphne::PrintOp>::Builtin;
    static const llvm::StringRef name;

    PrintBuiltin() : Builtin({1})
    {
    };
    mlir::daphne::PrintOp create(mlir::OpBuilder builder,
                                 mlir::Location &loc,
                                 mlir::ValueRange values) override;
};

struct RandBuiltin : public Builtin<mlir::daphne::RandMatrixOp>
{
    using Builtin<mlir::daphne::RandMatrixOp>::Builtin;
    static const llvm::StringRef name;

    RandBuiltin() : Builtin({2, 4})
    {
    };
    mlir::daphne::RandMatrixOp create(mlir::OpBuilder builder,
                                mlir::Location &loc,
                                mlir::ValueRange values) override;
};

struct TransposeBuiltin : public Builtin<mlir::daphne::TransposeOp>
{
    using Builtin<mlir::daphne::TransposeOp>::Builtin;
    static const llvm::StringRef name;

    TransposeBuiltin() : Builtin({1})
    {
    };
    mlir::daphne::TransposeOp create(mlir::OpBuilder builder,
                                     mlir::Location &loc,
                                     mlir::ValueRange values) override;
};

struct Builtins
{
    static antlrcpp::Any build(mlir::OpBuilder &builder,
                               mlir::Location &loc,
                               mlir::ValueRange values,
                               const std::string &name);
};

#endif //SRC_PARSER_DAPHNEDSL_BUILTINS_H
