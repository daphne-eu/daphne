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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_UNARYOPCODE_H
#define SRC_RUNTIME_LOCAL_KERNELS_UNARYOPCODE_H

#pragma once

// ****************************************************************************
// Enum for unary op codes and their names
// ****************************************************************************

enum class UnaryOpCode {
    // Arithmetic/general math.
    MINUS,
    ABS,
    SIGN, // signum (-1, 0, +1)
    SQRT,
    EXP,
    LN,
    // Trigonometric/hyperbolic.
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    SINH,
    COSH,
    TANH,
    // Rounding.
    FLOOR,
    CEIL,
    ROUND,
    // Comparison.
    ISNAN
};

/**
 * @brief Array of the "names" of the `UnaryOpCode`s.
 *
 * Must contain the same elements as `UnaryOpCode` in the same order,
 * such that we can obtain the name corresponding to a `UnaryOpCode` `opCode`
 * by `unary_op_codes[static_cast<int>(opCode)]`.
 */
static std::string_view unary_op_codes[] = {
    // Arithmetic/general math.
    "MINUS", "ABS", "SIGN", "SQRT", "EXP", "LN",
    // Trigonometric/hyperbolic.
    "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN", "SINH", "COSH", "TANH",
    // Rounding.
    "FLOOR", "CEIL", "ROUND",
    // Comparison.
    "ISNAN"
};

// ****************************************************************************
// Specification which unary ops should be supported on which value types
// ****************************************************************************

/**
 * @brief Template constant specifying if the given unary operation
 * should be supported on arguments of the given value type.
 *
 * @tparam op The unary operation.
 * @tparam VTRes The result value type.
 * @tparam VTArg The argument value type.
 */
template<UnaryOpCode op, typename VTRes, typename VTArg>
static constexpr bool supportsUnaryOp = false;

// Macros for concisely specifying which unary operations should be
// supported on which value types.

// Generates code specifying that the unary operation `Op` should be supported on
// the value type `VT` (for both the result and the argument, for simplicity).
#define SUPPORT(Op, VT) \
    template<> constexpr bool supportsUnaryOp<UnaryOpCode::Op, VT, VT> = true;

// Generates code specifying that all unary operations typically supported on
// numeric value types should be supported on the given value type `VT`
// (for both the result and the argument, for simplicity).
#define SUPPORT_NUMERIC(VT) \
    /* Arithmetic/general math. */ \
    SUPPORT(MINUS, VT) \
    SUPPORT(ABS  , VT) \
    SUPPORT(SIGN , VT) \
    SUPPORT(SQRT , VT) \
    SUPPORT(EXP  , VT) \
    SUPPORT(LN   , VT) \
    /* Trigonometric/hyperbolic. */ \
    SUPPORT(SIN , VT) \
    SUPPORT(COS , VT) \
    SUPPORT(TAN , VT) \
    SUPPORT(ASIN, VT) \
    SUPPORT(ACOS, VT) \
    SUPPORT(ATAN, VT) \
    SUPPORT(SINH, VT) \
    SUPPORT(COSH, VT) \
    SUPPORT(TANH, VT) \
    /* Rounding. */ \
    SUPPORT(FLOOR, VT) \
    SUPPORT(CEIL , VT) \
    SUPPORT(ROUND, VT) \
    /* Comparison */ \
    SUPPORT(ISNAN, VT)

// Concise specification of which unary operations should be supported on
// which value types.
SUPPORT_NUMERIC(double)
SUPPORT_NUMERIC(float)
SUPPORT_NUMERIC(int64_t)
SUPPORT_NUMERIC(int32_t)
SUPPORT_NUMERIC(int8_t)
SUPPORT_NUMERIC(uint64_t)
SUPPORT_NUMERIC(uint32_t)
SUPPORT_NUMERIC(uint8_t)

// Undefine helper macros.
#undef SUPPORT
#undef SUPPORT_NUMERIC

#endif //SRC_RUNTIME_LOCAL_KERNELS_UNARYOPCODE_H
