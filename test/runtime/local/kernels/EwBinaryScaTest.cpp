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

#include <runtime/local/kernels/EwBinarySca.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "EwBinarySca (" opName ")"
#define VALUE_TYPES double, uint32_t
#define STRING_TYPE const char*

template<BinaryOpCode opCode, typename VTArg, typename VTRes>
void checkEwBinarySca(VTArg lhs, VTArg rhs, VTRes exp) {
    if constexpr(std::is_same_v<VTRes, const char*>){
        CHECK(strcmp(EwBinarySca<opCode, VTRes, VTArg, VTArg>::apply(lhs, rhs, nullptr), exp) == 0);
        CHECK(strcmp(ewBinarySca<VTRes, VTArg, VTArg>(opCode, lhs, rhs, nullptr), exp) == 0);
    }else{
        CHECK(EwBinarySca<opCode, VTRes, VTArg, VTArg>::apply(lhs, rhs, nullptr) == exp);
        CHECK(ewBinarySca<VTRes, VTArg, VTArg>(opCode, lhs, rhs, nullptr) == exp);
    }
}
// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(0, 1, 1);
    checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(1, 2, 3);
}

TEMPLATE_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(0, 1, 0);
    checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(2, 3, 6);
}

TEMPLATE_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::DIV, VT, VT>(0, 3, 0);
    checkEwBinarySca<BinaryOpCode::DIV, VT, VT>(6, 3, 2);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(0, 0, 1);
    checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(3, 3, 1);
    checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(3, 5, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(3, 3, 0);
    checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(3, 5, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LT, VT, VT>(1, 1, 0);
    checkEwBinarySca<BinaryOpCode::LT, VT, VT>(1, 3, 1);
    checkEwBinarySca<BinaryOpCode::LT, VT, VT>(4, 2, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LE, VT, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::LE, VT, VT>(1, 3, 1);
    checkEwBinarySca<BinaryOpCode::LE, VT, VT>(4, 2, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GT, VT, VT>(1, 1, 0);
    checkEwBinarySca<BinaryOpCode::GT, VT, VT>(1, 3, 0);
    checkEwBinarySca<BinaryOpCode::GT, VT, VT>(4, 2, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GE, VT, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::GE, VT, VT>(1, 3, 0);
    checkEwBinarySca<BinaryOpCode::GE, VT, VT>(4, 2, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("eq - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::EQ, VT, int>("hi", "hi", 1);
    checkEwBinarySca<BinaryOpCode::EQ, VT, int>("hi", "bye", 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("neq - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::NEQ, VT, int>("hi", "hi", 0);
    checkEwBinarySca<BinaryOpCode::NEQ, VT, int>("hi", "bye", 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("lt - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LT, VT, int>("wow", "cool", 0);
    checkEwBinarySca<BinaryOpCode::LT, VT, int>("bye", "hi", 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("le - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LE, VT, int>("wow", "what", 0);
    checkEwBinarySca<BinaryOpCode::LE, VT, int>("bye", "hi", 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("gt - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GT, VT, int>("zebra", "what", 1);
    checkEwBinarySca<BinaryOpCode::GT, VT, int>("bye", "hi", 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("ge - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GE, VT, int>("zebra", "zebra", 1);
    checkEwBinarySca<BinaryOpCode::GE, VT, int>("bye", "hi", 0);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(2, 2, 2);
    checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(2, 3, 2);
    checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(3, 0, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(2, 2, 2);
    checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(2, 3, 3);
    checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(3, 0, 3);
}

TEMPLATE_TEST_CASE(TEST_NAME("min - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MIN, VT, const char*>("Antony", "John", "Antony");
}

TEMPLATE_TEST_CASE(TEST_NAME("max - string specific"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MAX, VT, const char*>("Antony", "John", "John");
}

// ****************************************************************************
// Logical
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 0,  0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 0,  1, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 1,  0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 1,  1, 1);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 0,  2, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 2,  0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 2,  2, 1);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>( 0, -2, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>(-2,  0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT, VT>(-2, -2, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 0,  0, 0);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 0,  1, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 1,  0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 1,  1, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 0,  2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 2,  0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 2,  2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>( 0, -2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>(-2,  0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT, VT>(-2, -2, 1);
}

// ****************************************************************************
// String
// ****************************************************************************
TEMPLATE_TEST_CASE(TEST_NAME("concat"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    VT lhs = "Hello";
    VT rhs = "World!";
    VT exp = "HelloWorld!";
    checkEwBinarySca<BinaryOpCode::CONCAT, VT, VT>(lhs, rhs, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("like"), TAG_KERNELS, STRING_TYPE) {
    using VT = TestType;
    VT lhs = "Heello";
    VT pattern = "H%l%l_";
    int exp = 1;
    checkEwBinarySca<BinaryOpCode::LIKE, VT, int>(lhs, pattern, exp);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    CHECK_THROWS(ewBinarySca<VT, VT, VT>(static_cast<BinaryOpCode>(999), 0, 0, nullptr));
}