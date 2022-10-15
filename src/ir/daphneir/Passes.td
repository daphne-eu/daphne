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

#ifndef SRC_IR_DAPHNEIR_PASSES_TD
#define SRC_IR_DAPHNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def DistributeComputations : FunctionPass<"distribute-computations"> {
    let constructor = "mlir::daphne::createDistributeComputationsPass()";
}

def DistributePipelines : FunctionPass<"distribute-pipelines"> {
    let constructor = "mlir::daphne::createDistributePipelinesPass()";
}

def Inference: FunctionPass<"inference"> {
    let constructor = "mlir::daphne::createInferencePass()";
}

def SelectMatrixRepresentations: FunctionPass<"select-matrix-representations"> {
    let constructor = "mlir::daphne::createSelectMatrixRepresentationsPass()";
}

def AdaptTypesToKernels: FunctionPass<"adapt-types-to-kernels"> {
    let constructor = "mlir::daphne::createAdaptTypesToKernelsPass()";
}

def ManageObjRefs : FunctionPass<"manage-obj-refs"> {
    let constructor = "mlir::daphne::createManageObjRefsPass()";
}

def PrintIR : FunctionPass<"print-ir"> {
    let constructor = "mlir::daphne::createPrintIRPass()";
}

def RewriteSqlOpPass : FunctionPass<"rewrite-sqlop"> {
    let constructor = "mlir::daphne::createRewriteSqlOpPass()";
}

def SpecializeGenericFunctionsPass : FunctionPass<"specialize-generic-funcs"> {
    let constructor = "mlir::daphne::createSpecializeGenericFunctionsPass()";
}

def WhileLoopInvariantCodeMotionPass : FunctionPass<"while-loop-invariant-code-motion"> {
    let constructor = "mlir::daphne::createWhileLoopInvariantCodeMotionPass()";
}

#endif // SRC_IR_DAPHNEIR_PASSES_TD