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

#ifndef SRC_IR_DAPHNEIR_DAPHNETYPEINFERENCETRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNETYPEINFERENCETRAITS_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Traits determining data type and value type separately.
// ****************************************************************************

// ============================================================================
// Data type
// ============================================================================

def DataTypeFromFirstArg: NativeOpTrait<"DataTypeFromFirstArg">;
def DataTypeFromArgs: NativeOpTrait<"DataTypeFromArgs">;
def DataTypeSca: NativeOpTrait<"DataTypeSca">;
def DataTypeMat: NativeOpTrait<"DataTypeMat">;
def DataTypeClm: NativeOpTrait<"DataTypeClm">;
def DataTypeFrm: NativeOpTrait<"DataTypeFrm">;
def DataTypeCol: NativeOpTrait<"DataTypeCol">;

// ============================================================================
// Value type
// ============================================================================

def ValueTypeCmp: NativeOpTrait<"ValueTypeCmp">;
def ValueTypeFromFirstArg: NativeOpTrait<"ValueTypeFromFirstArg">;
def ValueTypeFromThirdArg: NativeOpTrait<"ValueTypeFromThirdArg">;
def ValueTypeFromArgs: NativeOpTrait<"ValueTypeFromArgs">;
def ValueTypeFromArgsFP: NativeOpTrait<"ValueTypeFromArgsFP">;
def ValueTypeFromArgsInt: NativeOpTrait<"ValueTypeFromArgsInt">;
def ValueTypesConcat: NativeOpTrait<"ValueTypesConcat">;
def ValueTypeSI64: NativeOpTrait<"ValueTypeSI64">;
def ValueTypeSize: NativeOpTrait<"ValueTypeSize">;
def ValueTypeStr: NativeOpTrait<"ValueTypeStr">;

// ****************************************************************************
// Traits determining data type and value type together
// ****************************************************************************

def TypeFromFirstArg: NativeOpTrait<"TypeFromFirstArg">;

#endif //SRC_IR_DAPHNEIR_DAPHNETYPEINFERENCETRAITS_TD