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

#include <api/cli/StatusCode.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <exception>
#include <iostream>
#include <memory>

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace mlir;

OwningModuleRef
processModule(ModuleOp module)
{
    if(failed(verify(module))) {
        module.emitError("failed to verify the module right after parsing");
        return nullptr;
    }

    if (module) {
        //module->dump(); // print the DaphneIR representation
        PassManager pm(module->getContext());
        pm.addNestedPass<FuncOp>(daphne::createInsertDaphneContextPass());
        pm.addNestedPass<FuncOp>(daphne::createRewriteToCallKernelOpPass());
        pm.addPass(createLowerToCFGPass());
        pm.addPass(daphne::createLowerToLLVMPass());

        if (failed(pm.run(module))) {
            module->dump();
            module->emitError("module pass error");
            return nullptr;
        }
        return module;
    }
    return nullptr;
}

int
execJIT(OwningModuleRef & module)
{
    if (module) {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // An optimization pipeline to use within the execution engine.
        auto optPipeline = makeOptimizingTransformer(0, 0, nullptr);

        llvm::SmallVector<llvm::StringRef, 0> sharedLibRefs;
        // TODO Find these at run-time.
        sharedLibRefs.push_back("build/src/runtime/local/kernels/libAllKernels.so");
        registerLLVMDialectTranslation(*module->getContext());
        auto maybeEngine = ExecutionEngine::create(
                                                   module.get(), nullptr, optPipeline, llvm::CodeGenOpt::Level::Default,
                                                   sharedLibRefs, true, true, true);

        if (!maybeEngine) {
            llvm::errs() << "Failed to create JIT-Execution engine: "
                    << maybeEngine.takeError();
            return -1;
        }
        auto engine = maybeEngine->get();
        auto error = engine->invoke("main");
        if (error) {
            llvm::errs() << "JIT-Engine invokation failed: " << error;
            return -1;
        }
        return 0;
    }
    return -1;
}

int
main(int argc, char** argv)
{
    if (argc != 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        cout << "Usage: " << argv[0] << " FILE" << endl;
        exit(1);
    }

    // Parse command line arguments.
    string inputFile(argv[1]);

    // Create an MLIR context and load the required MLIR dialects.
    MLIRContext context;
    context.getOrLoadDialect<daphne::DaphneDialect>();
    context.getOrLoadDialect<StandardOpsDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();

    // Create an OpBuilder and an MLIR module and set the builder's insertion
    // point to the module's body, such that subsequently created DaphneIR
    // operations are inserted into the module.
    OpBuilder builder(&context);
    auto moduleOp = ModuleOp::create(builder.getUnknownLoc());
    auto * body = moduleOp.getBody();
    builder.setInsertionPoint(body, body->begin());

    // Parse the input file and generate the corresponding DaphneIR operations
    // inside the module, assuming DaphneDSL as the input format.
    DaphneDSLParser parser;
    try {
        parser.parseFile(builder, inputFile);
    }
    catch(std::exception & e) {
        std::cerr << "Parser error: " << e.what() << std::endl;
        return StatusCode::PARSER_ERROR;
    }
    
    // Further process the module, including optimization and lowering passes.
    OwningModuleRef module = processModule(moduleOp);
    
    // JIT-compile the module and execute it.
    // module->dump(); // print the LLVM IR representation
    execJIT(module);

    return StatusCode::SUCCESS;
}
