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

#include <parser/catalog/KernelCatalogParser.h>
#include "runtime/local/datastructures/Matrix.h"
#include <compiler/execution/DaphneIrExecutor.h>
#include <ir/daphneir/Daphne.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/context/DaphneContext.h>
#include <vector>
#include <string>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template <class DTRes, class DTArg> struct Recompile {
    static void apply(DTRes ** res, const DTArg ** arg, const char * mlirCode, size_t numInputs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTRes, class DTArg> void recompile(DTRes ** res, const DTArg ** arg, const char * mlirCode, size_t numInputs, DCTX(ctx)) {
    Recompile<DTRes, DTArg>::apply(res, arg, mlirCode, numInputs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ****************************************************************************
// Matrix
// ****************************************************************************
template <typename VTRes, typename VTArg> struct Recompile<Matrix<VTRes>, Matrix<VTArg>> {
    static void apply(Matrix<VTRes> ** res, const Matrix<VTArg> ** arg, const char * mlirCode, size_t numInputs, DCTX(ctx)) {
        
        auto cfg = ctx->getUserConfig();
        DaphneIrExecutor executor(true, cfg);

        KernelCatalog &kc = executor.getUserConfig().kernelCatalog;
        KernelCatalogParser kcp(executor.getContext());
        kcp.parseKernelCatalog("lib/catalog.json", kc);
        if (executor.getUserConfig().use_cuda)
            kcp.parseKernelCatalog("lib/CUDAcatalog.json", kc);

        llvm::StringRef mlirCodeRef(mlirCode);

        mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, executor.getContext());
        if (!module) {
            llvm::errs() << "Failed to parse source string.\n";
            return;
        }

        const char *functionName = "main";
        auto funcOp = module->lookupSymbol<mlir::func::FuncOp>(functionName);
        if (!funcOp) {
            llvm::errs() << "Function '" << functionName << "' not found in the module.\n";
            return;
        }
    
        auto recompileOpFuncTy = funcOp.getFunctionType();

        std::vector<void *> packedInputsOutputs;
        for (size_t i = 0; i < numInputs; ++i) {
            //arg[i]->increaseRefCounter();
            packedInputsOutputs.push_back(const_cast<void *>(static_cast<const void *>(arg[i])));
        }

        const size_t fixedNumOutputs = recompileOpFuncTy.getNumResults();
        for (size_t i = 0; i < fixedNumOutputs; ++i) {
            packedInputsOutputs.push_back(&(*res)[i]);
        }

        if (!executor.runPasses(module.get())) {
            llvm::errs() << "Module Pass Error.\n";
            return;
        }

        mlir::registerLLVMDialectTranslation(*module->getContext());
        auto engine = executor.createExecutionEngine(module.get());
        if (!engine) {
            llvm::errs() << "Failed to create JIT execution engine.\n";
            return;
        }

        auto error = engine->invokePacked(functionName, 
                                llvm::MutableArrayRef<void *>{packedInputsOutputs.data(), packedInputsOutputs.size()});
        if (error) {
            llvm::errs() << "JIT-Engine invocation failed: " << error << '\n';
            return;
        }
    }
};