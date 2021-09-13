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

def Inference: FunctionPass<"inference"> {
    let constructor = "mlir::daphne::createInferencePass()";
}

def InsertDaphneContext : FunctionPass<"insert-daphne-context"> {
    let constructor = "mlir::daphne::createInsertDaphneContextPass()";
}

def LowerToLLVM : FunctionPass<"lower-to-llvm"> {
    let constructor = "mlir::daphne::createLowerToLLVMPass()";
}

def PrintIR : FunctionPass<"print-ir"> {
    let constructor = "mlir::daphne::createPrintIRPass()";
}

def RewriteToCallKernelOp : FunctionPass<"rewrite-to-callkernelop"> {
    let constructor = "mlir::daphne::createRewriteToCallKernelOpPass()";
}

def VectorizeComputations : FunctionPass<"vectorize-computations"> {
    let constructor = "mlir::daphne::createVectorizeComputationsPass()";
}

def RewriteSqlOpPass : FunctionPass<"rewrite-to-sqlop"> {
    let constructor = "mlir::daphne::createRewriteSqlOpPass()";
}

def LowerRelationalAlgebraToDaphneOpPass : FunctionPass<"lower-rel_alg-to-daphne"> {
    let constructor = "mlir::daphne::createLowerRelationalAlgebraToDaphneOpPass()";
}

#endif // SRC_IR_DAPHNEIR_PASSES_TD