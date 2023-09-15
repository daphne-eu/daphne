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

#ifndef SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD
#define SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def InPlaceOpInterface : OpInterface<"InPlaceable"> {
    let description = [{
        Interface for operation that allow for in-place update of some operands.
    }];
  
    let methods = [
        InterfaceMethod<
        [{Returns the indices of potential operands that could be updated in-place.}],
        "std::vector<int>", "getInPlaceOperands", (ins)
        >,
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEUPDATEINPLACEENUM_TD
