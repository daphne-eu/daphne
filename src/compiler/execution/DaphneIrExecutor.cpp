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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>
#include "DaphneIrExecutor.h"

#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <filesystem>
#include <memory>
#include <utility>

DaphneIrExecutor::DaphneIrExecutor(bool selectMatrixRepresentations,
                                   DaphneUserConfig cfg)
    : selectMatrixRepresentations_(selectMatrixRepresentations),
    userConfig_(std::move(cfg)) {
    context_.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context_.getOrLoadDialect<mlir::StandardOpsDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context_.getOrLoadDialect<mlir::AffineDialect>();
    context_.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context_.getOrLoadDialect<mlir::linalg::LinalgDialect>();

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

bool DaphneIrExecutor::runPasses(mlir::ModuleOp module)
{
    // FIXME: operations in `template` functions (functions with unknown inputs) can't be verified
    //  as their type constraints are not met.
    //if (failed(mlir::verify(module))) {
        //module->emitError("failed to verify the module right after parsing");
        //return false;
    //}

    if (module) {
        // This flag is really useful to figure out why the lowering failed
        llvm::DebugFlag = userConfig_.debug_llvm;
        {
            mlir::PassManager pm(&context_);
            pm.enableVerifier(false);
            if(userConfig_.explain_parsing)
                pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing:"));
            pm.addPass(mlir::daphne::createSpecializeGenericFunctionsPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after specializing generic functions:"));
            if(failed(pm.run(module))) {
                module->dump();
                module->emitError("pass error for generic functions");
                return false;
            }
        }
        mlir::PassManager pm(&context_);
        pm.addPass(mlir::createCanonicalizerPass());
        if(userConfig_.explain_parsing_simplified)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing and some simplifications:"));
        pm.addPass(mlir::daphne::createRewriteSqlOpPass()); // calls SQL Parser
        if(userConfig_.explain_sql)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after SQL parsing:"));

        // TODO There is a cyclic dependency between (shape) inference and
        // constant folding (included in canonicalization), at the moment we
        // run only three iterations of both passes (see #173).
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInferencePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInferencePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInferencePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInferencePass());
        pm.addPass(mlir::createCanonicalizerPass());
        //pm.addPass(mlir::daphne::createPrintIRPass("IR after property inference"));

        if(selectMatrixRepresentations_) {
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createSelectMatrixRepresentationsPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after selecting matrix representation"));
        }

        if(userConfig_.explain_property_inference)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after property inference"));

        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createAdaptTypesToKernelsPass());
        if(userConfig_.explain_type_adaptation)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after type adaptation"));

#if 0
        if (userConfig_.use_distributed) {
            pm.addPass(mlir::daphne::createDistributeComputationsPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution"));
            pm.addPass(mlir::createCSEPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - CSE"));
            pm.addPass(mlir::createCanonicalizerPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - canonicalization"));
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createWhileLoopInvariantCodeMotionPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - WhileLICM"));
        }
#endif
        // TODO: add --explain argument
        if (userConfig_.codegen) {

            // pm.addPass(mlir::daphne::createPrintIRPass(
            //     "IR before LowerDenseMatrixPass"));
            pm.addPass(mlir::daphne::createLowerDenseMatrixPass());

            // pm.addNestedPass<mlir::FuncOp>(mlir::createLoopCoalescingPass());
            pm.addPass(mlir::daphne::createPrintIRPass(
                "IR after LowerDenseMatrixPass"));

            // pm.addPass(mlir::daphne::createMemRefTestPass());
            // pm.addPass(mlir::daphne::createPrintIRPass(
            //     "IR after MemRefTestPass"));

            // pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());

            // pm.addPass(mlir::daphne::createPrintIRPass(
            //     "IR after createLinalgBufferizePass"));
            // pm.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToLoopsPass());
            // // pm.addPass(mlir::createConvertLinalgToLLVMPass());
            //
            // pm.addPass(mlir::daphne::createPrintIRPass(
            //     "IR after createConvertLinalgToLoopsPass"));

            // pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
            // pm.addNestedPass<mlir::FuncOp>(mlir::createLoopCoalescingPass());
            // pm.addNestedPass<mlir::FuncOp>(mlir::createLoopFusionPass());

            pm.addPass(mlir::createLowerAffinePass());
            pm.addPass(mlir::daphne::createPrintIRPass(
                "IR after affine lowering"));
        }
        // For now, in order to use the distributed runtime we also require the vectorized engine to be enabled
        // to create pipelines. Therefore, *if* distributed runtime is enabled, we need to make a vectorization pass.
        if(userConfig_.use_vectorized_exec || userConfig_.use_distributed) {
            // TODO: add inference here if we have rewrites that could apply to vectorized pipelines due to smaller sizes
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createVectorizeComputationsPass());
            pm.addPass(mlir::createCanonicalizerPass());
        }
        if(userConfig_.explain_vectorized)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after vectorization"));
        
        if (userConfig_.use_distributed)
            pm.addPass(mlir::daphne::createDistributePipelinesPass());

        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createInsertDaphneContextPass(userConfig_));

#ifdef USE_CUDA
        if(userConfig_.use_cuda)
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createMarkCUDAOpsPass(userConfig_));
#endif

#ifdef USE_FPGAOPENCL
        if(userConfig_.use_fpgaopencl)
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createMarkFPGAOPENCLOpsPass(userConfig_));
#endif

        // Tidy up the IR before managing object reference counters with IncRefOp and DecRefOp.
        // This is important, because otherwise, an SSA value whose references are managed could
        // be cleared away by common subexpression elimination (CSE), while retaining its
        // IncRefOps/DecRefOps, which could lead to double frees etc.
        // pm.addPass(mlir::createCanonicalizerPass());
        // pm.addPass(mlir::createCSEPass());

        if(userConfig_.use_obj_ref_mgnt)
            pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createManageObjRefsPass());
        if(userConfig_.explain_obj_ref_mgnt)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after managing object references"));

        pm.addPass(mlir::daphne::createPrintIRPass("IR before kernel lowering"));
        pm.addNestedPass<mlir::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass());
        if(userConfig_.explain_kernels)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after kernel lowering"));

        pm.addPass(mlir::createLowerToCFGPass());
        // pm.addPass(mlir::daphne::createPrintIRPass("IR before LLVM lowering"));
        pm.addPass(mlir::daphne::createLowerToLLVMPass(userConfig_));
        if(userConfig_.explain_llvm)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after llvm lowering"));

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
        // TODO MSC: this enables loop unroll, tiling, unroll and jam, ..
        unsigned make_fast = 3;
        auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
        std::vector<llvm::StringRef> sharedLibRefs;
        // This next line adds to our Linux platform lock-in
        std::string daphne_executable_dir(std::filesystem::canonical("/proc/self/exe").parent_path());
        if(userConfig_.libdir.empty()) {
            sharedLibRefPaths.push_back(std::string(daphne_executable_dir + "/../lib/libAllKernels.so"));
            sharedLibRefs.emplace_back(sharedLibRefPaths.back());
        }
        else {
            sharedLibRefs.insert(sharedLibRefs.end(), userConfig_.library_paths.begin(), userConfig_.library_paths.end());
        }

#ifdef USE_CUDA
        if(userConfig_.use_cuda) {
            sharedLibRefPaths.push_back(std::string(daphne_executable_dir + "/../lib/libCUDAKernels.so"));
            sharedLibRefs.emplace_back(sharedLibRefPaths.back());
        }
#endif
 
#ifdef USE_FPGAOPENCL
        if(userConfig_.use_fpgaopencl) {
            if(userConfig_.libdir.empty()) {
                std::string fpgaKernelsPath(std::string(daphne_executable_dir + "/../lib/libFPGAOPENCLKernels.so"));
                sharedLibRefs.push_back(fpgaKernelsPath);
            }
        }
#endif
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
