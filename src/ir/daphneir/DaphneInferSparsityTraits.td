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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYTRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYTRAITS_TD

include "mlir/IR/OpBase.td"

// Sets the sparsity to 1.0 (no zero values)
// Usually there are edge cases where an operation could still have a zero value, but as
// it won't for all the other cases estimating with 1.0 is fine
def CompletelyDense : NativeOpTrait<"CompletelyDense">;

// Passes the input sparsity to the output
def SparsityRemains : NativeOpTrait<"SparsityRemains">;

// Sets the sparsity to -1.0 if the sparsity will be unknown after an operation
def SparsityUnknown : NativeOpTrait<"SparsityUnknown">; 

class SparsityFromIthScalar<int i> : ParamNativeOpTrait<"SparsityFromIthScalar", !cast<string>(i)>;
class SparsityFromIthArg<int i> : ParamNativeOpTrait<"SparsityFromIthArg", !cast<string>(i)>;
def SparsityFromArg : SparsityFromIthArg<0>;

// ****************************************************************************
// Traits determining sparsity for elementwise ops
// ****************************************************************************

// if both elements are zero, the result is zero
def EwSparseIfBoth : NativeOpTrait<"EwSparseIfBoth">;
// if either element is zero, the result is zero
def EwSparseIfEither : NativeOpTrait<"EwSparseIfEither">;
// if all input elements are completely sparse, the resulting sparsity is 0.0 (works for 1 or 2 operands)
def SparseIfAllInputSparse : NativeOpTrait<"SparseIfAllInputSparse">;
// if all input elements are completely dense, the resulting sparsity is 1.0 (works for 1 or 2 operands)
def DenseIfAllInputDense : NativeOpTrait<"DenseIfAllInputDense">;
// if all input elements are either completely sparse or completely dense, the resulting sparsity is 0.0 or 1.0 respectively (works for 1 or 2 operands)
def SparsityRemainsIfAllInputOneOrZero : NativeOpTrait<"SparsityRemainsIfAllInputOneOrZero">;

#endif //SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYTRAITS_TD