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

#ifndef SRC_IR_DAPHNEIR_DAPHNESHAPEINFERENCETRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNESHAPEINFERENCETRAITS_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Traits determining #rows or #cols separately.
// ****************************************************************************

def OneRow : NativeOpTrait<"OneRow">;
def OneCol : NativeOpTrait<"OneCol">;
class NumRowsFromIthScalar<int i> : ParamNativeOpTrait<"NumRowsFromIthScalar", !cast<string>(i)>;
class NumColsFromIthScalar<int i> : ParamNativeOpTrait<"NumColsFromIthScalar", !cast<string>(i)>;
class NumRowsFromIthArg<int i> : ParamNativeOpTrait<"NumRowsFromIthArg", !cast<string>(i)>;
class NumColsFromIthArg<int i> : ParamNativeOpTrait<"NumColsFromIthArg", !cast<string>(i)>;
class NumRowsFromIthArgNumCols<int i> : ParamNativeOpTrait<"NumRowsFromIthArgNumCols", !cast<string>(i)>;
class NumColsFromIthArgNumRows<int i> : ParamNativeOpTrait<"NumColsFromIthArgNumRows", !cast<string>(i)>;
def NumRowsFromArg : NumRowsFromIthArg<0>;
def NumColsFromArg : NumColsFromIthArg<0>;
def NumRowsFromArgNumCols : NumRowsFromIthArgNumCols<0>;
def NumColsFromArgNumRows : NumColsFromIthArgNumRows<0>;
def NumRowsFromAllArgs : NativeOpTrait<"NumRowsFromAllArgs">;
def NumColsFromAllArgs : NativeOpTrait<"NumColsFromAllArgs">;
def NumRowsFromSumOfAllArgs : NativeOpTrait<"NumRowsFromSumOfAllArgs">;
def NumColsFromSumOfAllArgs : NativeOpTrait<"NumColsFromSumOfAllArgs">;

// ****************************************************************************
// Traits determining #rows and #cols together.
// ****************************************************************************

class ShapeFromIthArg<int i> : ParamNativeOpTrait<"ShapeFromIthArg", !cast<string>(i)>;
def ShapeFromArg : ShapeFromIthArg<0>;
def ShapeEwBinary: NativeOpTrait<"ShapeEwBinary">;

#endif //SRC_IR_DAPHNEIR_DAPHNESHAPEINFERENCETRAITS_TD