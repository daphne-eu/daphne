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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferShapeOpInterface : OpInterface<"InferShape"> {
    let description = [{
        Interface to infer the shape(s) of the data object(s)
        returned by an operation.
    }];

    let methods = [
        InterfaceMethod<
                "Infer the shape(s) the output data object(s).",
                "std::vector<std::pair<ssize_t, ssize_t>>", "inferShape", (ins)
        >
    ];
}

def InferNumRowsOpInterface : OpInterface<"InferNumRows"> {
    let description = [{
        Interface to infer the number of rows of the single data object
        returned by an operation.
    }];

    let methods = [
        InterfaceMethod<
                "Infer the number of rows of the single output data object.",
                "ssize_t", "inferNumRows", (ins)
        >
    ];
}

def InferNumColsOpInterface : OpInterface<"InferNumCols"> {
    let description = [{
        Interface to infer the number of columns of the single data object
        returned by an operation.
    }];

    let methods = [
        InterfaceMethod<
                "Infer the number of columns of the single output data object.",
                "ssize_t", "inferNumCols", (ins)
        >
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERSHAPEOPINTERFACE_TD