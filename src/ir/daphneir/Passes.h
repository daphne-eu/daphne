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

#ifndef SRC_IR_DAPHNEIR_PASSES_H
#define SRC_IR_DAPHNEIR_PASSES_H

#pragma once

#include <api/cli/DaphneUserConfig.h>

#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir::daphne {
    struct InferenceConfig {
        InferenceConfig(bool partialInferenceAllowed,
                        bool typeInference,
                        bool shapeInference,
                        bool frameLabelInference,
                        bool sparsityInference);
        bool partialInferenceAllowed;
        bool typeInference;
        bool shapeInference;
        bool frameLabelInference;
        bool sparsityInference;
    };

    // alphabetically sorted list of passes
    std::unique_ptr<Pass> createAdaptTypesToKernelsPass();
    std::unique_ptr<Pass> createDistributeComputationsPass();
    std::unique_ptr<Pass> createDistributePipelinesPass();
    std::unique_ptr<Pass> createInferencePass(InferenceConfig cfg = {false, true, true, true, true});
    std::unique_ptr<Pass> createInsertDaphneContextPass(const DaphneUserConfig& cfg);
    std::unique_ptr<Pass> createProfilingPass();
    std::unique_ptr<Pass> createLowerToLLVMPass(const DaphneUserConfig& cfg);
    std::unique_ptr<Pass> createManageObjRefsPass();
    std::unique_ptr<Pass> createPhyOperatorSelectionPass();
    std::unique_ptr<Pass> createPrintIRPass(std::string message = "");
    std::unique_ptr<Pass> createRewriteSqlOpPass();
    std::unique_ptr<Pass> createRewriteToCallKernelOpPass();
    std::unique_ptr<Pass> createSelectMatrixRepresentationsPass();
    std::unique_ptr<Pass> createSpecializeGenericFunctionsPass(const DaphneUserConfig& cfg);
    std::unique_ptr<Pass> createVectorizeComputationsPass();
    std::unique_ptr<Pass> createWhileLoopInvariantCodeMotionPass();
    std::unique_ptr<Pass> createFlagUpdateInPlacePass();
#ifdef USE_CUDA
    std::unique_ptr<Pass> createMarkCUDAOpsPass(const DaphneUserConfig& cfg);
#endif

#ifdef USE_FPGAOPENCL
    std::unique_ptr<Pass> createMarkFPGAOPENCLOpsPass(const DaphneUserConfig& cfg);
#endif

#define GEN_PASS_REGISTRATION
#include "ir/daphneir/Passes.h.inc"
} // namespace mlir::daphne

#endif //SRC_IR_DAPHNEIR_PASSES_H
