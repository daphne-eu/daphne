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

#pragma once

// ****************************************************************************
// Enum for binary op codes and their names
// ****************************************************************************

enum class BinaryOpCode {
    // Arithmetic.
    ADD,  // addition
    SUB,  // subtraction
    MUL,  // multiplication
    DIV,  // division
    POW,  // to the power of
    MOD,  // modulus
    LOG,  // logarithm
    // Comparisons.
    EQ,   // equal
    NEQ,  // not equal
    LT,   // less than
    LE,   // less equal
    GT,   // greater than
    GE,   // greater equal
    // Min/max.
    MIN,
    MAX,
    // Logical.
    AND,
    OR,
    // Bitwise.
    BITWISE_AND,
    // Strings.
    CONCAT
};

/**
 * @brief Array of the "names" of the `BinaryOpCode`s.
 *
 * Must contain the same elements as `BinaryOpCode` in the same order,
 * such that we can obtain the name corresponding to a `BinaryOpCode` `opCode`
 * by `binary_op_codes[static_cast<int>(opCode)]`.
 */
static std::string_view binary_op_codes[] = {
    // Arithmetic.
    "ADD", "SUB", "MUL", "DIV", "POW", "MOD", "LOG",
    // Comparisons.
    "EQ", "NEQ", "LT", "LE", "GT", "GE",
    // Min/max.
    "MIN", "MAX",
    // Logical.
    "AND", "OR",
    // Bitwise.
    "BITWISE_AND",
    // Strings.
    "CONCAT"
};

// ****************************************************************************
// Specification which binary ops should be supported on which value types
// ****************************************************************************

/**
 * @brief Template constant specifying if the given binary operation
 * should be supported on arguments of the given value types.
 *
 * @tparam VTRes The result value type.
 * @tparam VTLhs The left-hand-side argument value type.
 * @tparam VTRhs The right-hand-side argument value type.
 * @tparam op The binary operation.
 */
template<BinaryOpCode op, typename VTRes, typename VTLhs, typename VTRhs>
static constexpr bool supportsBinaryOp = false;

// Macros for concisely specifying which binary operations should be
// supported on which value types.

// Generates code specifying that the binary operation `Op` should be supported on
// the value type `VT` (for the result and the two arguments, for simplicity).
#define SUPPORT(Op, VT) \
    template<> constexpr bool supportsBinaryOp<BinaryOpCode::Op, VT, VT, VT> = true;

// Generates code specifying that all binary operations of a certain category should be
// supported on the given value type `VT` (for the result and the two arguments, for simplicity).
#define SUPPORT_ARITHMETIC(VT) \
    /* Arithmetic. */ \
    SUPPORT(ADD, VT) \
    SUPPORT(SUB, VT) \
    SUPPORT(MUL, VT) \
    SUPPORT(DIV, VT) \
    SUPPORT(POW, VT) \
    SUPPORT(MOD, VT) \
    SUPPORT(LOG, VT)
#define SUPPORT_EQUALITY(VT) \
    /* Comparisons. */ \
    SUPPORT(EQ , VT) \
    SUPPORT(NEQ, VT)
#define SUPPORT_COMPARISONS(VT) \
    /* Comparisons. */ \
    SUPPORT(LT, VT) \
    SUPPORT(LE, VT) \
    SUPPORT(GT, VT) \
    SUPPORT(GE, VT) \
    /* Min/max. */ \
    SUPPORT(MIN, VT) \
    SUPPORT(MAX, VT)
#define SUPPORT_LOGICAL(VT) \
    /* Logical. */ \
    SUPPORT(AND, VT) \
    SUPPORT(OR , VT)
#define SUPPORT_BITWISE(VT) \
    /* Bitwise. */ \
    SUPPORT(BITWISE_AND, VT)

// Generates code specifying that all binary operations typically supported on a certain
// category of value types should be supported on the given value type `VT`
// (for the result and the two arguments, for simplicity).
#define SUPPORT_NUMERIC_FP(VT) \
    SUPPORT_ARITHMETIC(VT) \
    SUPPORT_EQUALITY(VT) \
    SUPPORT_COMPARISONS(VT) \
    SUPPORT_LOGICAL(VT)
#define SUPPORT_NUMERIC_INT(VT) \
    SUPPORT_ARITHMETIC(VT) \
    SUPPORT_EQUALITY(VT) \
    SUPPORT_COMPARISONS(VT) \
    SUPPORT_LOGICAL(VT) \
    SUPPORT_BITWISE(VT)

// Concise specification of which binary operations should be supported on
// which value types.
SUPPORT_NUMERIC_FP(double)
SUPPORT_NUMERIC_FP(float)
SUPPORT_NUMERIC_INT(int64_t)
SUPPORT_NUMERIC_INT(int32_t)
SUPPORT_NUMERIC_INT(int8_t)
SUPPORT_NUMERIC_INT(uint64_t)
SUPPORT_NUMERIC_INT(uint32_t)
SUPPORT_NUMERIC_INT(uint8_t)
template<> constexpr bool supportsBinaryOp<BinaryOpCode::CONCAT, const char *, const char *, const char *> = true;
template<> constexpr bool supportsBinaryOp<BinaryOpCode::EQ, int64_t, const char *, const char *> = true;

// Undefine helper macros.
#undef SUPPORT
#undef SUPPORT_ARITHMETIC
#undef SUPPORT_EQUALITY
#undef SUPPORT_COMPARISONS
#undef SUPPORT_LOGICAL
#undef SUPPORT_BITWISE
#undef SUPPORT_NUMERIC_FP
#undef SUPPORT_NUMERIC_INT
