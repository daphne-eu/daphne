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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferTypesOpInterface : OpInterface<"InferTypes"> {
    let description = [{
        Interface to infer the result types of an operation.
    }];

    let methods = [
        InterfaceMethod<
                "Infer the result types of the operation.",
                "std::vector<mlir::Type>", "inferTypes", (ins)
        >
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERTYPESOPINTERFACE_TD
