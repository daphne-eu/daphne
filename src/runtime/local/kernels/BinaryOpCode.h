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
};

static std::string_view binary_op_codes[] = {"ADD", "SUB", "MUL", "DIV", "POW", "MOD", "LOG", "EQ", "NEQ", "LT", "LE",
        "GT", "GE", "MIN", "MAX", "AND", "OR", "BITWISE_AND"};
