/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <tags.h>

#include <catch.hpp>

#include <parser/daphnedsl/DaphneDSLParser.h>
#include "ir/daphneir/Daphne.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/AsmState.h>
#include "mlir/Parser/Parser.h"
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <ir/daphneir/Passes.h>

#include <iostream>
#include <memory>

#include <cstring>

const std::string dirPath = "test/api/cli/parser/";

/// This testcase tests both parsing of a simple DML file, while also checking if the printing and parsing of
/// dialect operations and types is compatible.
TEST_CASE("Parse file in DML, write and re-read as DaphneIR", TAG_PARSER)
{
    std::string daphneIrCode;
    std::string daphneIRCodeMatRepr;
    {
        mlir::MLIRContext context;
        context.getOrLoadDialect<mlir::daphne::DaphneDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();

        mlir::OpBuilder builder(&context);
        auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
        auto *body = moduleOp.getBody();
        builder.setInsertionPoint(body, body->begin());

        DaphneDSLParser parser;
        parser.parseFile(builder, dirPath + "simple.daphne");

        llvm::raw_string_ostream stream(daphneIrCode);
        moduleOp.print(stream);
        
        // Print IR after SelectMatrixRepresentationsPass
        mlir::PassManager passManager(&context);
        passManager.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInferencePass());
        passManager.addPass(mlir::createCanonicalizerPass());
        passManager.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createSelectMatrixRepresentationsPass());
        
        REQUIRE(failed(passManager.run(moduleOp)) == false);
        
        llvm::raw_string_ostream streamMatRepr(daphneIRCodeMatRepr);
        moduleOp.print(streamMatRepr);
    }

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::parseSourceString<mlir::ModuleOp>(daphneIrCode, &context));
    REQUIRE(module);

    std::string newCode;
    llvm::raw_string_ostream stream(newCode);
    module->print(stream);

    REQUIRE(daphneIrCode == newCode);
        
    // Parse after SelectMatrixRepresentationsPass
    mlir::OwningOpRef<mlir::ModuleOp> moduleMatReprPass(mlir::parseSourceString<mlir::ModuleOp>(daphneIRCodeMatRepr, &context));
    REQUIRE(moduleMatReprPass);

    std::string newCodeMatRepr;
    llvm::raw_string_ostream streamMatRepr(newCodeMatRepr);
    moduleMatReprPass->print(streamMatRepr);

    REQUIRE(daphneIRCodeMatRepr == newCodeMatRepr);
}
