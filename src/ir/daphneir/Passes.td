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

def DistributeComputations : Pass<"distribute-computations", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDistributeComputationsPass()";
}

def DistributePipelines : Pass<"distribute-pipelines", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDistributePipelinesPass()";
}

def Inference: Pass<"inference", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createInferencePass()";
}

def SelectMatrixRepresentations: Pass<"select-matrix-representations", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createSelectMatrixRepresentationsPass()";
}

def AdaptTypesToKernels: Pass<"adapt-types-to-kernels", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createAdaptTypesToKernelsPass()";
}

def ManageObjRefs : Pass<"manage-obj-refs", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createManageObjRefsPass()";
}

def PrintIR : Pass<"print-ir", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createPrintIRPass()";
}

def RewriteSqlOpPass : Pass<"rewrite-sqlop", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createRewriteSqlOpPass()";
}

def WhileLoopInvariantCodeMotionPass : Pass<"while-loop-invariant-code-motion", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createWhileLoopInvariantCodeMotionPass()";
}

def AggAllLoweringPass : Pass<"lower-agg", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createAggAllOpLoweringPass()";
}

def MatMulOpLoweringPass : Pass<"lower-mm", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createMatMulOpLoweringPass()";
}

def DaphneOpsOptPass : Pass<"opt-daphne", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDaphneOptPass()";
}

def MapOpLoweringPass: Pass<"lower-map", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createMapOpLoweringPass()";
}

def LowerEwOpPass: Pass<"lower-ew", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createEwOpLoweringPass()";
}


#endif // SRC_IR_DAPHNEIR_PASSES_TD
