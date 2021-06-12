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
    mlir::OpBuilder & builder;
    
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
     * @brief A DaphneIR `Matrix` with value type DaphneIR `Size`.
     */
    mlir::Type matrixOfSizeType;
    
    /**
     * @brief Get a `daphne::MatrixType` with the given value type.
     * @param vt
     * @return 
     */
    mlir::daphne::MatrixType matrixOf(mlir::Type vt) {
        return mlir::daphne::MatrixType::get(builder.getContext(), vt);
    }
    
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
    
    ParserUtils(mlir::OpBuilder & builder)
    :
            builder(builder),
            sizeType(builder.getIndexType()),
            boolType(builder.getI1Type()),
            seedType(builder.getIntegerType(64, false)), // TODO is this really UI, or I?
            matrixOfSizeType(static_cast<mlir::Type>(mlir::daphne::MatrixType::get(builder.getContext(), sizeType)))
    {
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
        if(v.getType() == t)
            return v;
        return builder.create<mlir::daphne::CastOp>(v.getLoc(), t, v);
    }
    
    mlir::Value castSizeIf(mlir::Value v) {
        return castIf(sizeType, v);
    }
    
    mlir::Value castBoolIf(mlir::Value v) {
        return castIf(boolType, v);
    }
    
    mlir::Value castSeedIf(mlir::Value v) {
        return castIf(seedType, v);
    }
    
    // ************************************************************************
    // Misc
    // ************************************************************************
    
    mlir::Value valueOrError(antlrcpp::Any a) {
        if(a.is<mlir::Value>())
            return a.as<mlir::Value>();
        throw std::runtime_error("something was expected to be an mlir::Value, but it was none");
    }
    
};

#endif //SRC_PARSER_PARSERUTILS_H