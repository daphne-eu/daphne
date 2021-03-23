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

#include "DaphneLexer.h"
#include "DaphneParser.h"
#include "parser/daphnedsl/MLIRGenVisitors.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "antlr4-runtime.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <iostream>
#include <memory>

#include <cstdlib>
#include <cstring>

using namespace std;
using namespace mlir;

OwningModuleRef
processModule(MLIRContext & context, daphne_antlr::DaphneParser::FileContext *file)
{
    OpBuilder builder(&context);
    mlir_gen::FileVisitor visitor(builder);
    auto module = visitor.visitFile(file).as<ModuleOp>();
    
    if(failed(verify(module))) {
        module.emitError("failed to verify the module right after parsing");
        return nullptr;
    }

    if (module) {
        //module->dump(); // print the DaphneIR representation
        PassManager pm(module->getContext());

        pm.addPass(daphne::createRewriteToCallKernelOpPass());
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
        sharedLibRefs.push_back("build/src/runtime/local/kernels/libPrintKernels.so");
        sharedLibRefs.push_back("build/src/runtime/local/kernels/libLinAlgKernels.so");
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

    string inputFile(argv[1]);

    MLIRContext context;
    context.getOrLoadDialect<daphne::DaphneDialect>();
    context.getOrLoadDialect<StandardOpsDialect>();

    ifstream stream;
    stream.open(inputFile, ios::in);
    if (!stream.is_open()) {
        cerr << "Could not open file \"" << inputFile << '"' << endl;
        return 1;
    }

    antlr4::ANTLRInputStream input(stream);
    input.name = inputFile;
    daphne_antlr::DaphneLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    daphne_antlr::DaphneParser parser(&tokens);

    auto *file = parser.file();

    OwningModuleRef module = processModule(context, file);
    // module->dump(); // print the LLVM IR representation
    execJIT(module);

    return 0;
}
