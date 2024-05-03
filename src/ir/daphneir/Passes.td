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
  let description = [{
        This pass lowers the daphne::MatMulOp to an affine loop structure instead of using pre-compiled Kernels.
        Square and non-square matrices can be lowered.
        The default result is the naive loop nest over lhs rows, rhs columns and a body loop.
        If a vector size is specified, MatMuls on sizes that are cleanly divided by the vector size use affine vector instructions.
        If tiling is enabled a tiling scheme inspired by https://github.com/bondhugula/llvm-project/blob/hop/mlir/docs/HighPerfCodeGen.md 
        is employed. The general scheme turns the (i, j, k) loops into an affine schedule of eight loops 
        (j / NC, k / KC, i / MC, (j mod NC) / NR, (i mod MC) / MR, k mod KC, (j mod NC) mod NR, (i mod MC) mod MR) specified by five tile sizes.
        Optionally the last two loops (j mod NR, i mod MR) are partially unrolled and 'jammed' into their parent loops, if NR and MR 
        cleanly divide the matrix size. Finally an optional unroll factor is applied to the resulting inner most loop.

        Example 1: daphne-opt --lower-mm
        Input:
        ```
        module {
        func.func @main() {
        %0 = "daphne.constant"() {value = 10 : index} : () -> index
            %1 = "daphne.constant"() {value = false} : () -> i1
            %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
            %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
            %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
            %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
            %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
            "daphne.return"() : () -> ()
        }
        }
        ```
        Output: 
        ```
        module {
        func.func @main() {
            %alloc = memref.alloc() : memref<10x10xf64>
            %0 = "daphne.constant"() {value = 10 : index} : () -> index
            %1 = "daphne.constant"() {value = false} : () -> i1
            %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
            %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
            %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
            %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
            %6 = "daphne.convertDenseMatrixToMemRef"(%4) : (!daphne.Matrix<10x10xf64>) -> memref<10x10xf64>
            %7 = "daphne.convertDenseMatrixToMemRef"(%5) : (!daphne.Matrix<10x10xf64>) -> memref<10x10xf64>
            %cst = arith.constant 0.000000e+00 : f64
            affine.for %arg0 = 0 to 10 {
            affine.for %arg1 = 0 to 10 {
                affine.store %cst, %alloc[%arg0, %arg1] : memref<10x10xf64>
            }
            }
            affine.for %arg0 = 0 to 10 {
            affine.for %arg1 = 0 to 10 {
                affine.for %arg2 = 0 to 10 {
                %9 = affine.load %6[%arg0, %arg2] : memref<10x10xf64>
                %10 = affine.load %7[%arg2, %arg1] : memref<10x10xf64>
                %11 = affine.load %alloc[%arg0, %arg1] : memref<10x10xf64>
                %12 = llvm.intr.fma(%9, %10, %11)  : (f64, f64, f64) -> f64
                affine.store %12, %alloc[%arg0, %arg1] : memref<10x10xf64>
                }
            }
            }
            %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %alloc : memref<10x10xf64> -> memref<f64>, index, index, index, index, index
            %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<10x10xf64> -> index
            %8 = "daphne.convertMemRefToDenseMatrix"(%intptr, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1) : (index, index, index, index, index, index) -> !daphne.Matrix<10x10xf64>
            "daphne.return"() : () -> ()
        }
        }
        ```
    Example 2: daphne-opt --lower-mm="matmul_fixed_tile_sizes=2,2,2,2,2 matmul_use_fixed_tile_sizes=true matmul_tile=true matmul_unroll_jam_factor=2"
    Input:
    ```
    module {
    func.func @main() {
        %0 = "daphne.constant"() {value = 10 : index} : () -> index
        %1 = "daphne.constant"() {value = false} : () -> i1
        %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
        %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
        %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
        %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
        %6 = "daphne.matMul"(%4, %5, %1, %1) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>, i1, i1) -> !daphne.Matrix<10x10xf64>
        "daphne.return"() : () -> ()
    }
    }
    ```
    Output:
    ```
    #map = affine_map<(d0) -> (d0)>
    #map1 = affine_map<(d0) -> (d0 + 2)>
    #map2 = affine_map<(d0) -> (d0 + 1)>
    module {
    func.func @main() {
        %alloc = memref.alloc() : memref<10x10xf64>
        %0 = "daphne.constant"() {value = 10 : index} : () -> index
        %1 = "daphne.constant"() {value = false} : () -> i1
        %2 = "daphne.constant"() {value = 3.000000e+00 : f64} : () -> f64
        %3 = "daphne.constant"() {value = 5.000000e+00 : f64} : () -> f64
        %4 = "daphne.fill"(%3, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
        %5 = "daphne.fill"(%2, %0, %0) : (f64, index, index) -> !daphne.Matrix<10x10xf64>
        %6 = "daphne.convertDenseMatrixToMemRef"(%4) : (!daphne.Matrix<10x10xf64>) -> memref<10x10xf64>
        %7 = "daphne.convertDenseMatrixToMemRef"(%5) : (!daphne.Matrix<10x10xf64>) -> memref<10x10xf64>
        %cst = arith.constant 0.000000e+00 : f64
        affine.for %arg0 = 0 to 10 {
        affine.for %arg1 = 0 to 10 {
            affine.store %cst, %alloc[%arg0, %arg1] : memref<10x10xf64>
        }
        }
        affine.for %arg0 = 0 to 10 step 2 {
        affine.for %arg1 = 0 to 10 step 2 {
            affine.for %arg2 = 0 to 10 step 2 {
            affine.for %arg3 = #map(%arg0) to #map1(%arg0) step 2 {
                affine.for %arg4 = #map(%arg2) to #map1(%arg2) step 2 {
                affine.for %arg5 = #map(%arg4) to #map1(%arg4) {
                    %9 = affine.load %6[%arg5, %arg1] : memref<10x10xf64>
                    %10 = affine.load %7[%arg1, %arg3] : memref<10x10xf64>
                    %11 = affine.load %alloc[%arg5, %arg3] : memref<10x10xf64>
                    %12 = llvm.intr.fma(%9, %10, %11)  : (f64, f64, f64) -> f64
                    affine.store %12, %alloc[%arg5, %arg3] : memref<10x10xf64>
                    %13 = affine.apply #map2(%arg1)
                    %14 = affine.load %6[%arg5, %13] : memref<10x10xf64>
                    %15 = affine.load %7[%13, %arg3] : memref<10x10xf64>
                    %16 = affine.load %alloc[%arg5, %arg3] : memref<10x10xf64>
                    %17 = llvm.intr.fma(%14, %15, %16)  : (f64, f64, f64) -> f64
                    affine.store %17, %alloc[%arg5, %arg3] : memref<10x10xf64>
                    %18 = affine.apply #map2(%arg3)
                    %19 = affine.load %6[%arg5, %arg1] : memref<10x10xf64>
                    %20 = affine.load %7[%arg1, %18] : memref<10x10xf64>
                    %21 = affine.load %alloc[%arg5, %18] : memref<10x10xf64>
                    %22 = llvm.intr.fma(%19, %20, %21)  : (f64, f64, f64) -> f64
                    affine.store %22, %alloc[%arg5, %18] : memref<10x10xf64>
                    %23 = affine.apply #map2(%arg1)
                    %24 = affine.load %6[%arg5, %23] : memref<10x10xf64>
                    %25 = affine.load %7[%23, %18] : memref<10x10xf64>
                    %26 = affine.load %alloc[%arg5, %18] : memref<10x10xf64>
                    %27 = llvm.intr.fma(%24, %25, %26)  : (f64, f64, f64) -> f64
                    affine.store %27, %alloc[%arg5, %18] : memref<10x10xf64>
                }
                }
            }
            }
        }
        }
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %alloc : memref<10x10xf64> -> memref<f64>, index, index, index, index, index
        %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<10x10xf64> -> index
        %8 = "daphne.convertMemRefToDenseMatrix"(%intptr, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1) : (index, index, index, index, index, index) -> !daphne.Matrix<10x10xf64>
        "daphne.return"() : () -> ()
    }
    }
    ```
  }];
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
           "Set the factor by which to unroll the inner most loop inside the lowered matrix multiplication as a last step. "
           "No unrolling by default.">,
    Option<"matmul_unroll_jam_factor", "matmul_unroll_jam_factor", "int",
           /*default=*/"4",
           "Set the factor by which to unroll jam the two inner most loop inside the lowered matrix multiplication."
           "The default value is affines default value.">,
    Option<"matmul_num_vec_registers", "matmul_num_vec_registers", "int",
           /*default=*/"16",
           "Set the number of vector registers, used during automatic tiling when lowering matrix multiplication."
           "The default value is affines default value.">,
    Option<"matmul_invert_loops", "matmul_invert_loops", "bool",
           /*default=*/"false",
           "Enable inverting of the inner two loops in the matrix multiplication as a fallback option, if tiling is not possible or deactivated. "
           "Switched off by default.">,
           
  ];
}

def LowerEwOpPass: Pass<"lower-ew", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createEwOpLoweringPass()";
}


#endif // SRC_IR_DAPHNEIR_PASSES_TD
