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

#ifndef SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def VECTOR_SPLIT_NONE : I64EnumAttrCase<"NONE", 0>;
def VECTOR_SPLIT_ROWS : I64EnumAttrCase<"ROWS", 1>;
def VECTOR_SPLIT_COLS : I64EnumAttrCase<"COLS", 2>;

def VectorSplitAttr : I64EnumAttr<"VectorSplit", "", [VECTOR_SPLIT_NONE, VECTOR_SPLIT_ROWS, VECTOR_SPLIT_COLS]> {
    let cppNamespace = "::mlir::daphne";
}

def VECTOR_COMBINE_ROWS : I64EnumAttrCase<"ROWS", 1>;
def VECTOR_COMBINE_COLS : I64EnumAttrCase<"COLS", 2>;
def VECTOR_COMBINE_ADD : I64EnumAttrCase<"ADD", 3>;
def VECTOR_COMBINE_MAX : I64EnumAttrCase<"MAX", 4>;
def VECTOR_COMBINE_MIN : I64EnumAttrCase<"MIN", 5>;

def VectorCombineAttr : I64EnumAttr<"VectorCombine", "", [VECTOR_COMBINE_ROWS, VECTOR_COMBINE_COLS, VECTOR_COMBINE_ADD, VECTOR_COMBINE_MAX, VECTOR_COMBINE_MIN]> {
    let cppNamespace = "::mlir::daphne";
}

def VectorizableOpInterface : OpInterface<"Vectorizable"> {
    let description = [{
        Enables the computation in the form of vectors.
    }];

    let methods = [
        InterfaceMethod<"Get the vector split kind for each input.",
                        "std::vector<std::vector<daphne::VectorSplit>>", "getVectorSplits", (ins)>,
        InterfaceMethod<"Get the vector combine kind for each output.",
                        "std::vector<std::vector<daphne::VectorCombine>>", "getVectorCombines", (ins)>,
        InterfaceMethod<"Create values for #rows and #cols of each output. -1 for dynamic/unknown.",
                        "std::vector<std::vector<std::pair<Value, Value>>>", "createOpsOutputSizes", (ins "mlir::OpBuilder&":$builder)>,
        // TODO: for complex operations (non element-wise) where the computation per vector is not equal to the operation
        //  itself on the whole input, we will require a new method generating the operations in the pipeline. This is
        //  the same behaviour as with `Distributable` Ops, and therefore combining them might make sense.
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEVECTORIZABLEOPINTERFACE_TD
