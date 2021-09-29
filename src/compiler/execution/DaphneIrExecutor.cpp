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

#include <parser/daphnedsl/DaphneDSLParser.h>
#include <parser/sql/SQLParser.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>
#include "DaphneIrExecutor.h"

#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <exception>
#include <memory>

#include <iostream>

DaphneIrExecutor::DaphneIrExecutor(bool distributed, bool vectorized)
: distributed_(distributed), vectorized_(vectorized)
{
    context_.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context_.getOrLoadDialect<mlir::StandardOpsDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

bool DaphneIrExecutor::runPasses(mlir::ModuleOp module)
{
    if (failed(mlir::verify(module))) {
        module->emitError("failed to verify the module right after parsing");
        return false;
    }

    if (module) {
        mlir::PassManager pm(&context_);

        // This flag is really useful to figure out why the lowering failed
        //llvm::DebugFlag = true;
        // pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing:"));
        pm.addPass(mlir::daphne::createRewriteSqlOpPass());   //calls SQL Parser
        // pm.addPass(mlir::daphne::createPrintIRPass("IR after SQL parsing:"));

        pm.addPass(mlir::daphne::createLowerRelationalAlgebraToDaphneOpPass());
        if (distributed_) {
            pm.addPass(mlir::daphne::createDistributeComputationsPass());
        }
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInferencePass());
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInsertDaphneContextPass());
        if(vectorized_) {
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createVectorizeComputationsPass());
            // TODO: this can be moved outside without problem, should we?
            pm.addPass(mlir::createCanonicalizerPass());
        }
        pm.addPass(mlir::createCSEPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass());
        //pm.addPass(mlir::daphne::createPrintIRPass("IR after kernel lowering"));

        pm.addPass(mlir::createLowerToCFGPass());
        pm.addPass(mlir::daphne::createLowerToLLVMPass());
        // pm.addPass(mlir::daphne::createPrintIRPass("IR after llvm lowering"));

        if (failed(pm.run(module))) {
            module->dump();
            module->emitError("module pass error");
            return false;
        }
        return true;
    }
    return false;
}

std::unique_ptr<mlir::ExecutionEngine> DaphneIrExecutor::createExecutionEngine(mlir::ModuleOp module)
{
    if (module) {
        // An optimization pipeline to use within the execution engine.
        auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);

        llvm::SmallVector<llvm::StringRef, 1> sharedLibRefs;
        // TODO Find these at run-time.
        sharedLibRefs.push_back("build/src/runtime/local/kernels/libAllKernels.so");
        registerLLVMDialectTranslation(context_);
        // module.dump();
        auto maybeEngine = mlir::ExecutionEngine::create(
            module, nullptr, optPipeline, llvm::CodeGenOpt::Level::Default,
            sharedLibRefs, true, true, true);

        if (!maybeEngine) {
            llvm::errs() << "Failed to create JIT-Execution engine: "
                         << maybeEngine.takeError();
            return nullptr;
        }
        return std::move(maybeEngine.get());
    }
    return nullptr;
}
