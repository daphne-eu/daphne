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

#ifndef SRC_PARSER_DAPHNEDSL_DAPHNEDSLBUILTINS_H
#define SRC_PARSER_DAPHNEDSL_DAPHNEDSLBUILTINS_H

#include <parser/ParserUtils.h>
#include <runtime/local/io/FileMetaData.h>

#include "antlr4-runtime.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>
#include <vector>

#include <cstdlib>

/**
 * @brief Utility for creating DaphneIR operations for DaphneDSL built-in
 * functions.
 */
class DaphneDSLBuiltins {
    
    /**
     * @brief The OpBuilder used to generate DaphneIR operations.
     */
    mlir::OpBuilder & builder;
    
    /**
     * @brief General utilities for parsing to DaphneIR.
     */
    ParserUtils utils;
    
    // ************************************************************************
    // Checking number of arguments
    // ************************************************************************
    
    static void checkNumArgsExact(const std::string & func, size_t numArgs, size_t numArgsExact);
    
    static void checkNumArgsBetween(const std::string & func, size_t numArgs, size_t numArgsMin, size_t numArgsMax);
    
    static void checkNumArgsMin(const std::string & func, size_t numArgs, size_t numArgsMin);
    
    static void checkNumArgsEven(const std::string & func, size_t numArgs);
    
    // ************************************************************************
    // Creating similar DaphneIR operations
    // ************************************************************************
    
    template<class NumOp>
    mlir::Value createNumOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class UnaryOp>
    mlir::Value createUnaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class BinaryOp>
    mlir::Value createBinaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class RowAggOp, class ColAggOp>
    mlir::Value createRowOrColAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class GrpAggOp>
    mlir::Value createGrpAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class AllAggOp, class RowAggOp, class ColAggOp, class GrpAggOp>
    mlir::Value createAnyAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class CumAggOp>
    mlir::Value createCumAggOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    mlir::Value createQuantizeOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    template<class BindOp>
    mlir::Value createBindOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class TheOp>
    mlir::Value createSameTypeUnaryOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    mlir::Value createTriOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args, bool upper);
    
    template<class SetOp>
    mlir::Value createSetOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
    template<class JoinOp>
    mlir::Value createJoinOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    mlir::Value createAffineFwdOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    mlir::Value createBatchNorm2dTestFwdOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    mlir::ResultRange createConv2dFwdOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    template<class PoolOp>
    mlir::ResultRange createPoolFwdOp(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);

    // ************************************************************************
    // Other utilities
    // ************************************************************************
    
    FileMetaData getFileMetaData(const std::string & func, mlir::Value filename);
    
    // ************************************************************************
    
public:
    
    explicit DaphneDSLBuiltins(mlir::OpBuilder & builder) : builder(builder), utils(builder) {
        //
    };

    antlrcpp::Any build(mlir::Location loc, const std::string & func, const std::vector<mlir::Value> & args);
    
};

#endif //SRC_PARSER_DAPHNEDSL_DAPHNEDSLBUILTINS_H
