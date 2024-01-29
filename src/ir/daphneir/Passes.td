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

def DaphneOpsOptPass : Pass<"opt-daphne", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDaphneOptPass()";
}

def MapOpLoweringPass: Pass<"lower-map", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createMapOpLoweringPass()";
}

def MatMulOpLoweringPass: Pass<"lower-mm", "::mlir::ModuleOp"> {
  let summary = "Lower the MatMulOp acting on Daphne matrices to an affine loop structure.";
  let dependentDialects = ["vector::VectorDialect", "mlir::LLVM::LLVMDialect", "mlir::AffineDialect",
                            "mlir::memref::MemRefDialect"];
  let constructor = "mlir::daphne::createMatMulOpLoweringPass()";
  let options = [
    ListOption<"matmul_fixed_tile_sizes", "matmul_fixed_tile_sizes", "unsigned",
               "Specify up to 5 tile sizes (MR, NR, KC, MC, NC) to use when tiling loops in the matrix multipliation. Tiling needs to be enabled separately. The resulting tiling is (j / NC, k / KC, i / MC, j / NR, i / MR, k mod KC, j mod NR, i mod MR), where the innermost two loops are unrolled jammed by a factor of up to 4.">,
    Option<"matmul_tile", "matmul_tile", "bool",
           /*default=*/"false",
           "Enable tiling of the loops in the matrix multiplication. "
           "Switched off by default.">,
    Option<"matmul_use_fixed_tile_sizes", "matmul_use_fixed_tile_sizes", "bool",
           /*default=*/"false",
           "Enable tiling of the loops in the matrix multiplication with separately specified fixed tile sizes. "
           "Switched off by default. In daphne (not daphne-opt) is activated by simply providing fixed sizes.">,
    Option<"matmul_vec_size_bits", "matmul_vec_size_bits", "int",
           /*default=*/"0",
           "Set the maximum bit size of the vector instructions to use during matrix multiplication if their use is possible. 0 disables vectorization. "
           "Switched off by default.">,
    Option<"matmul_unroll_factor", "matmul_unroll_factor", "int",
           /*default=*/"1",
           "Set the factor by which to unroll the inner most loop inside the lowered matrix multiplication. "
           "No unrolling by default.">
           
  ];
}

def LowerEwOpPass: Pass<"lower-ew", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createEwOpLoweringPass()";
}


#endif // SRC_IR_DAPHNEIR_PASSES_TD
