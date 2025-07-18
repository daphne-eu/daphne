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

#ifndef SRC_PARSER_PARSERUTILS_H
#define SRC_PARSER_PARSERUTILS_H

#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <util/ErrorHandler.h>

#include "antlr4-runtime.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <stdexcept>

/**
 * @brief General utilities for parsing to DaphneIR.
 *
 *
 */
class ParserUtils {
    /**
     * The OpBuilder used to generate DaphneIR operations.
     */
    mlir::OpBuilder &builder;

  public:
    // ************************************************************************
    // `mlir::Type`s corresponsing to the types in `DaphneTypes.td`
    // ************************************************************************

    /**
     * @brief The `mlir::Type` denoted by `Size` in DaphneIR.
     */
    const mlir::Type sizeType;

    /**
     * @brief The `mlir::Type` denoted by `BoolScalar` in DaphneIR.
     */
    const mlir::Type boolType;

    /**
     * @brief The `mlir::Type` denoted by `Seed` in DaphneIR.
     */
    const mlir::Type seedType;

    /**
     * @brief The `mlir::Type` denoted by `StrScalar` in DaphneIR.
     */
    const mlir::Type strType;

    /**
     * @brief A DaphneIR `Matrix` with value type DaphneIR `Size`.
     */
    mlir::Type matrixOfSizeType;

    /**
     * @brief The placeholder for an unknown type.
     */
    mlir::Type unknownType;

    /**
     * @brief Get a `daphne::MatrixType` with the given value type.
     * @param vt
     * @return
     */
    mlir::daphne::MatrixType matrixOf(mlir::Type vt) { return mlir::daphne::MatrixType::get(builder.getContext(), vt); }

    /**
     * @brief Get a `daphne::MatrixType` with the type of the given value as
     * the value type.
     * @param vt
     * @return
     */
    mlir::daphne::MatrixType matrixOf(mlir::Value v) {
        return mlir::daphne::MatrixType::get(builder.getContext(), v.getType());
    }

    // ************************************************************************
    // Constructor
    // ************************************************************************

    ParserUtils(mlir::OpBuilder &builder)
        : builder(builder), sizeType(builder.getIndexType()), boolType(builder.getI1Type()),
          seedType(builder.getIntegerType(64, true)), strType(mlir::daphne::StringType::get(builder.getContext())),
          matrixOfSizeType(static_cast<mlir::Type>(mlir::daphne::MatrixType::get(builder.getContext(), sizeType))),
          unknownType(mlir::daphne::UnknownType::get(builder.getContext())) {
        // nothing to do
    }

    // ************************************************************************
    // Casting if necessary
    // ************************************************************************

    /**
     * @brief Wraps the given `Value` in a `CastOp` if it does not have the
     * given `Type`.
     *
     * @param loc A location.
     * @param t The expected type.
     * @param v The value.
     * @return `v` if it has type `t`, otherwise a `CastOp` of `v` to `t`.
     */
    mlir::Value castIf(mlir::Type t, mlir::Value v) {
        if (v.getType() == t)
            return v;
        return builder.create<mlir::daphne::CastOp>(v.getLoc(), t, v);
    }

    mlir::Value castSizeIf(mlir::Value v) { return castIf(sizeType, v); }

    mlir::Value castBoolIf(mlir::Value v) { return castIf(boolType, v); }

    mlir::Value castSeedIf(mlir::Value v) { return castIf(seedType, v); }

    mlir::Value castStrIf(mlir::Value v) { return castIf(strType, v); }

    mlir::Value castUI8If(mlir::Value v) { return castIf(builder.getIntegerType(8, false), v); }

    mlir::Value castUI32If(mlir::Value v) { return castIf(builder.getIntegerType(32, false), v); }

    mlir::Value castUI64If(mlir::Value v) { return castIf(builder.getIntegerType(64, false), v); }

    mlir::Value castSI8If(mlir::Value v) { return castIf(builder.getIntegerType(8, true), v); }

    mlir::Value castSI32If(mlir::Value v) { return castIf(builder.getIntegerType(32, true), v); }

    mlir::Value castSI64If(mlir::Value v) { return castIf(builder.getIntegerType(64, true), v); }

    mlir::Value castF64If(mlir::Value v) { return castIf(builder.getF64Type(), v); }

    // ************************************************************************
    // Type parsing
    // ************************************************************************

    mlir::Type getValueTypeByName(const std::string &name) {
        if (name == "f64")
            return builder.getF64Type();
        if (name == "f32")
            return builder.getF32Type();
        if (name == "si64")
            return builder.getIntegerType(64, true);
        if (name == "si32")
            return builder.getIntegerType(32, true);
        if (name == "si8")
            return builder.getIntegerType(8, true);
        if (name == "ui64")
            return builder.getIntegerType(64, false);
        if (name == "ui32")
            return builder.getIntegerType(32, false);
        if (name == "ui8")
            return builder.getIntegerType(8, false);
        if (name == "str")
            return strType;
        if (name == "bool")
            return boolType;
        throw std::runtime_error("unsupported value type: " + name);
    }

    // ************************************************************************
    // Misc
    // ************************************************************************

    /**
     * @brief Returns the given result of an ANTLR visitor function as an
     * `mlir::Value` or throws an informative error if it is not a single value.
     *
     * @param loc The location in the input file (e.g., DaphneDSL script or SQL
     * query).
     * @param a The result returned by an ANTLR visitor function (e.g., by
     * `visit()` or `visitXYZ()`).
     */
    mlir::Value valueOrError(mlir::Location loc, antlrcpp::Any a) {
        if (a.is<mlir::Value>())
            return a.as<mlir::Value>();
        if (a.isNull())
            // Typically happens when a *user-defined* function with zero return
            // values is used in a place where exactly one value is required
            // (e.g., in an expression).
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              "the expression was expected to return a single value, but it "
                                              "returns no value");
        if (a.is<mlir::Operation *>())
            // Typically happens when a *built-in* function with zero return
            // values is used in a place where exactly one value is required
            // (e.g., in an expression).
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              "the expression was expected to return a single value, but it "
                                              "returns no value");
        if (a.is<mlir::ResultRange>()) {
            // Typically happens when a *built-in or user-defined* function with
            // more than one return values is used in a place where exactly one
            // value is required (e.g., in an expression).
            auto rr = a.as<mlir::ResultRange>();
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              "the expression was expected to return a single value, but it "
                                              "returns " +
                                                  std::to_string(rr.size()) + " values");
        }
        // This should never happen. If it happens, that indicates that this
        // function is called on an invalid argument or that some visitor
        // returns a wrong result type.
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "the expression was expected to return a single value, but it is "
                                          "something unexpected");
    }

    mlir::Type typeOrError(antlrcpp::Any a) {
        if (a.is<mlir::Type>())
            return a.as<mlir::Type>();
        throw std::runtime_error("something was expected to be an mlir::Type, but it was none");
    }

    /**
     * @brief Utility function for getting the file location of the token
     * @param start Start token of this rule (usually you want to use
     * `ctx->start`)
     * @return mlir location representing the position of the token in the file
     */
    mlir::Location getLoc(antlr4::Token *start) {
        return mlir::FileLineColLoc::get(builder.getStringAttr(start->getTokenSource()->getSourceName()),
                                         start->getLine(), start->getCharPositionInLine());
    }

    /**
     * @brief Creates an unique symbol for function symbol names by appending an
     * unique id.
     * @param functionName the function name
     * @return the unique function name, due to an unique id
     */
    std::string getUniqueFunctionSymbol(const std::string &functionName) {
        static unsigned functionUniqueId = 0;
        return functionName + "-" + std::to_string(++functionUniqueId);
    }
};

#endif // SRC_PARSER_PARSERUTILS_H
