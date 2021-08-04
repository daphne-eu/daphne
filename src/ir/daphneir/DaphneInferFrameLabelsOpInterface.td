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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERFRAMELABELSOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERFRAMELABELSOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferFrameLabelsOpInterface : OpInterface<"InferFrameLabels"> {
    let description = [{
        Interface to infer the column labels of a frame returned by an 
        operation. This information is useful for type inference.
    }];

    let methods = [
        InterfaceMethod<
                "Infer the column labels of the output frame.",
                "void", "inferFrameLabels", (ins)
        >
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERFRAMELABELSOPINTERFACE_TD