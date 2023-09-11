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
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <filesystem>
#include <memory>
#include <utility>

DaphneIrExecutor::DaphneIrExecutor(bool selectMatrixRepresentations, DaphneUserConfig cfg) : userConfig_(std::move(cfg)),
        selectMatrixRepresentations_(selectMatrixRepresentations) {
    // register loggers
    if(userConfig_.log_ptr != nullptr)
        userConfig_.log_ptr->registerLoggers();

    context_.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

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
            // TODO Enable the verifier for all passes where it is possible.
            // Originally, it was only turned off for the SpecializeGenericFunctionsPass.
            pm.enableVerifier(false);

            if(userConfig_.explain_parsing)
                pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing:"));

            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
            if(userConfig_.explain_parsing_simplified)
                pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing and some simplifications:"));

            pm.addPass(mlir::daphne::createRewriteSqlOpPass()); // calls SQL Parser
            if(userConfig_.explain_sql)
                pm.addPass(mlir::daphne::createPrintIRPass("IR after SQL parsing:"));

            pm.addPass(mlir::daphne::createSpecializeGenericFunctionsPass(userConfig_));
            if(userConfig_.explain_property_inference)
                pm.addPass(mlir::daphne::createPrintIRPass("IR after inference:"));

            if(failed(pm.run(module))) {
                module->dump();
                module->emitError("module pass error");
                return false;
            }
        }
        mlir::PassManager pm(&context_);

        // Note that property inference and canonicalization have already been done
        // in the SpecializeGenericFunctionsPass, so actually, it's not necessary
        // here anymore.
        // TODO There is a cyclic dependency between (shape) inference and
        // constant folding (included in canonicalization), at the moment we
        // run only three iterations of both passes (see #173).
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInferencePass());
        pm.addPass(mlir::createCanonicalizerPass());

        if(selectMatrixRepresentations_)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createSelectMatrixRepresentationsPass());
        if(userConfig_.explain_select_matrix_repr)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after selecting matrix representations:"));

        if(userConfig_.use_phy_op_selection) {
            pm.addPass(mlir::daphne::createPhyOperatorSelectionPass());
            pm.addPass(mlir::createCSEPass());
        }
        if(userConfig_.explain_phy_op_selection)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after selecting physical operators:"));

        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createAdaptTypesToKernelsPass());
        if(userConfig_.explain_type_adaptation)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after type adaptation:"));

#if 0
        if (userConfig_.use_distributed) {
            pm.addPass(mlir::daphne::createDistributeComputationsPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution:"));
            pm.addPass(mlir::createCSEPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - CSE:"));
            pm.addPass(mlir::createCanonicalizerPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - canonicalization:"));
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createWhileLoopInvariantCodeMotionPass());
            //pm.addPass(mlir::daphne::createPrintIRPass("IR after distribution - WhileLICM:"));
        }
#endif
        
        // For now, in order to use the distributed runtime we also require the vectorized engine to be enabled
        // to create pipelines. Therefore, *if* distributed runtime is enabled, we need to make a vectorization pass.
        if(userConfig_.use_vectorized_exec || userConfig_.use_distributed) {
            // TODO: add inference here if we have rewrites that could apply to vectorized pipelines due to smaller sizes
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createVectorizeComputationsPass());
            pm.addPass(mlir::createCanonicalizerPass());
        }
        if(userConfig_.explain_vectorized)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after vectorization:"));
        
        if (userConfig_.use_distributed)
            pm.addPass(mlir::daphne::createDistributePipelinesPass());

        if (userConfig_.enable_profiling)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createProfilingPass());

        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInsertDaphneContextPass(userConfig_));

#ifdef USE_CUDA
        if(userConfig_.use_cuda)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createMarkCUDAOpsPass(userConfig_));
#endif

#ifdef USE_FPGAOPENCL
        if(userConfig_.use_fpgaopencl)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createMarkFPGAOPENCLOpsPass(userConfig_));
#endif

        // Tidy up the IR before managing object reference counters with IncRefOp and DecRefOp.
        // This is important, because otherwise, an SSA value whose references are managed could
        // be cleared away by common subexpression elimination (CSE), while retaining its
        // IncRefOps/DecRefOps, which could lead to double frees etc.
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        if(userConfig_.use_obj_ref_mgnt)
            pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createManageObjRefsPass());
        if(userConfig_.explain_obj_ref_mgnt)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after managing object references:"));

        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass());
        if(userConfig_.explain_kernels)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after kernel lowering:"));

        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
        pm.addPass(mlir::daphne::createLowerToLLVMPass(userConfig_));
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        if(userConfig_.explain_llvm)
            pm.addPass(mlir::daphne::createPrintIRPass("IR after llvm lowering:"));

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
            sharedLibRefPaths.push_back(std::string(daphne_executable_dir + "/../lib/libFPGAOPENCLKernels.so"));
            sharedLibRefs.emplace_back(sharedLibRefPaths.back());
        }
#endif
        registerLLVMDialectTranslation(context_);
        // module.dump();
        mlir::ExecutionEngineOptions options;
        options.llvmModuleBuilder = nullptr;
        options.transformer = optPipeline;
        options.jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Default;
        options.sharedLibPaths = llvm::ArrayRef<llvm::StringRef>(sharedLibRefs);
        options.enableObjectDump = true;
        options.enableGDBNotificationListener = true;
        options.enablePerfNotificationListener = true;
        auto maybeEngine = mlir::ExecutionEngine::create(module, options);

        if (!maybeEngine) {
            llvm::errs() << "Failed to create JIT-Execution engine: "
                         << maybeEngine.takeError();
            return nullptr;
        }
        return std::move(maybeEngine.get());
    }
    return nullptr;
}
