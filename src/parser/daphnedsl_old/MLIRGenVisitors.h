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

#ifndef SRC_PARSER_DAPHNEDSL_MLIRGENVISITORS_H
#define SRC_PARSER_DAPHNEDSL_MLIRGENVISITORS_H

#include "DaphneLexer.h"
#include "DaphneParser.h"
#include "DaphneParserBaseVisitor.h"

#include "antlr4-runtime.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ScopedHashTable.h"

using daphne_antlr::DaphneParser;
using daphne_antlr::DaphneParserBaseVisitor;

namespace mlir_gen {

class DaphneMlirVisitor : public DaphneParserBaseVisitor {
  protected:
    mlir::OpBuilder &builder;

    mlir::Location getLocMLIR(antlr4::Token *token);
    static llvm::Optional<llvm::APFloat>
    parseFloatLiteral(mlir::Location loc, llvm::StringRef floatLiteral);
    static llvm::Optional<llvm::APInt>
    parseIntegerLiteral(mlir::Location loc, llvm::StringRef decimalLiteral,
                        unsigned int bitWidth);

  public:
    explicit DaphneMlirVisitor(mlir::OpBuilder &builder);

    antlrcpp::Any visitFloatType(DaphneParser::FloatTypeContext *ctx) override;
    antlrcpp::Any
    visitIntegerType(DaphneParser::IntegerTypeContext *ctx) override;

    static std::string parseStringLiteral(llvm::StringRef floatLiteral);
};

struct FileVisitor : public DaphneMlirVisitor {
    using DaphneMlirVisitor::DaphneMlirVisitor;
    antlrcpp::Any visitFile(DaphneParser::FileContext *ctx) override;
};

struct ItemVisitor : public DaphneMlirVisitor {
    using DaphneMlirVisitor::DaphneMlirVisitor;
    antlrcpp::Any visitItem(DaphneParser::ItemContext *ctx) override;
};

class FunctionVisitor : public DaphneMlirVisitor {
    // FIXME: this maps the variable name to a pair of variable name and value.
    //  This is only done to ensure that the string, the stringref is referring
    //  to exists.
    std::map<std::string, mlir::Value> symbolTable;

    __attribute_warn_unused_result__ mlir::LogicalResult
    declareVar(std::string name, mlir::Value value,
               bool allowShadowing = false);

  public:
    using DaphneMlirVisitor::DaphneMlirVisitor;
    antlrcpp::Any visitFunction(DaphneParser::FunctionContext *ctx) override;
    antlrcpp::Any
    visitFunctionArgs(DaphneParser::FunctionArgsContext *ctx) override;
    antlrcpp::Any
    visitFunctionArg(DaphneParser::FunctionArgContext *ctx) override;
    antlrcpp::Any visitLiteralExpression(
        DaphneParser::LiteralExpressionContext *ctx) override;
    antlrcpp::Any visitLiteralExpressionRule(
        DaphneParser::LiteralExpressionRuleContext *ctx) override;
    antlrcpp::Any
    visitBlockStatement(DaphneParser::BlockStatementContext *ctx) override;
    antlrcpp::Any visitAssignmentExpression(
        DaphneParser::AssignmentExpressionContext *ctx) override;
    antlrcpp::Any
    visitLetStatement(DaphneParser::LetStatementContext *ctx) override;
    antlrcpp::Any
    visitCallExpression(DaphneParser::CallExpressionContext *ctx) override;
    antlrcpp::Any visitIdentifierExpression(
        DaphneParser::IdentifierExpressionContext *ctx) override;
    antlrcpp::Any visitStatement(DaphneParser::StatementContext *ctx) override;
    antlrcpp::Any visitExpressionStatement(
        DaphneParser::ExpressionStatementContext *ctx) override;
    antlrcpp::Any
    visitParameters(DaphneParser::ParametersContext *ctx) override;
    antlrcpp::Any visitParameter(DaphneParser::ParameterContext *ctx) override;
    antlrcpp::Any visitArithmeticExpression(
        DaphneParser::ArithmeticExpressionContext *ctx) override;
    antlrcpp::Any visitGroupedExpression(
        DaphneParser::GroupedExpressionContext *ctx) override;
    antlrcpp::Any
    visitMatrixLiteral(DaphneParser::MatrixLiteralContext *ctx) override;
    antlrcpp::Any visitMatrixLiteralElements(
        DaphneParser::MatrixLiteralElementsContext *ctx) override;
};

class MatrixLiteral {
    bool initialized = false;
    long rows = -1;
    long cols = -1;
    mlir::Type elementType = nullptr;
    std::vector<llvm::APFloat> linearizedFloatData;
    std::vector<llvm::APInt> linearizedIntData;

  public:
    void addData(mlir::Location &loc, mlir::OpBuilder &builder,
                 antlrcpp::Any data);
    [[nodiscard]] long getRows() const { return rows; }
    [[nodiscard]] long getCols() const { return cols; }
    [[nodiscard]] mlir::Type getElementType() const { return elementType; }
    [[nodiscard]] std::vector<llvm::APFloat> getLinFloatData() const {
        return linearizedFloatData;
    }
    [[nodiscard]] std::vector<llvm::APInt> getLinIntData() const {
        return linearizedIntData;
    }
};
} // namespace mlir_gen
#endif // SRC_PARSER_DAPHNEDSL_MLIRGENVISITORS_H