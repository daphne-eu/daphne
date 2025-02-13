/*
 * Copyright 2025 The DAPHNE Consortium
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

 #pragma once

 // ****************************************************************************
 // Enum for comparison op codes and their names
 // ****************************************************************************
 
 enum class CmpOpCode {
     EQ, NEQ, GT, GE, LT, LE
 };
 
 /**
  * @brief Array of the "names" of the `CmpOpCode`s.
  *
  * Must contain the same elements as `CmpOpCode` in the same order,
  * such that we can obtain the name corresponding to a `CmpOpCode` `opCode`
  * by `comparison_op_codes[static_cast<int>(opCode)]`.
  */
 static std::string_view comparison_op_codes[] = {"EQ", "NEQ", "GT", "GE", "LT", "LE"};
 
 // ****************************************************************************
 // Specification which comparison ops should be supported on which value types
 // ****************************************************************************
 
 /**
  * @brief Template constant specifying if the given comparison operation
  * should be supported on arguments of the given value types.
  *
  * @tparam VTRes The result value type.
  * @tparam VTLhs The left-hand-side argument value type.
  * @tparam VTRhs The right-hand-side argument value type.
  * @tparam op The binary operation.
  */
 template <CmpOpCode op, typename VTRes, typename VTLhs, typename VTRhs>
 static constexpr bool supportsCmpOp = false;
 
 // Macros for concisely specifying which comparison operations should be
 // supported on which value types.
 
 // Generates code specifying that the comparison operation `Op` should be supported
 // on the value type `VT` (for the result and the two arguments, for
 // simplicity).
 #define SUPPORT(Op, VT) template <> constexpr bool supportsCmpOp<CmpOpCode::Op, VT, VT, VT> = true;

 #define SUPPORT_ALL_VTS(Op) \
    SUPPORT(Op, double) \
    SUPPORT(Op, float) \
    SUPPORT(Op, int64_t) \
    SUPPORT(Op, int32_t) \
    SUPPORT(Op, int8_t) \
    SUPPORT(Op, uint64_t) \
    SUPPORT(Op, uint32_t) \
    SUPPORT(Op, uint8_t) \
    SUPPORT(Op, bool) \
    SUPPORT(Op, std::string)

SUPPORT_ALL_VTS(EQ);
SUPPORT_ALL_VTS(NEQ);
SUPPORT_ALL_VTS(GT);
SUPPORT_ALL_VTS(GE);
SUPPORT_ALL_VTS(LT);
SUPPORT_ALL_VTS(LE);
 
 // Undefine helper macros.
 #undef SUPPORT
 #undef SUPPORT_ALL_VTS