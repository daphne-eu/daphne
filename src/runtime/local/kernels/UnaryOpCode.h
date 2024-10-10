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
    "ISNAN"};
