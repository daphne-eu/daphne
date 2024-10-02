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

template <BinaryOpCode opCode, typename VT> void checkEwBinarySca(VT lhs, VT rhs, VT exp) {
    CHECK(EwBinarySca<opCode, VT, VT, VT>::apply(lhs, rhs, nullptr) == exp);
    CHECK(ewBinarySca<VT, VT, VT>(opCode, lhs, rhs, nullptr) == exp);
}

template<BinaryOpCode opCode>
void checkEwBinarySca(std::string lhs, std::string rhs, int exp) {
    CHECK(EwBinarySca<opCode, int64_t, std::string, std::string>::apply(lhs, rhs, nullptr) == exp);
    CHECK(ewBinarySca<int64_t, std::string, std::string>(opCode, lhs, rhs, nullptr) == exp);
}

template<BinaryOpCode opCode>
void checkEwBinarySca(FixedStr16 lhs, FixedStr16 rhs, int exp) {
    CHECK(EwBinarySca<opCode, int64_t, FixedStr16, FixedStr16>::apply(lhs, rhs, nullptr) == exp);
    CHECK(ewBinarySca<int64_t, FixedStr16, FixedStr16>(opCode, lhs, rhs, nullptr) == exp);
}

template<typename VT>
void checkEwBinarySca(VT lhs, VT rhs, std::string exp) {
    CHECK(EwBinarySca<BinaryOpCode::CONCAT, std::string, VT, VT>::apply(lhs, rhs, nullptr) == exp);
    CHECK(ewBinarySca<std::string, VT, VT>(BinaryOpCode::CONCAT, lhs, rhs, nullptr) == exp);
}

// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::ADD, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::ADD, VT>(0, 1, 1);
    checkEwBinarySca<BinaryOpCode::ADD, VT>(1, 2, 3);
}

TEMPLATE_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MUL, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::MUL, VT>(0, 1, 0);
    checkEwBinarySca<BinaryOpCode::MUL, VT>(2, 3, 6);
}

TEMPLATE_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::DIV, VT>(0, 3, 0);
    checkEwBinarySca<BinaryOpCode::DIV, VT>(6, 3, 2);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::EQ, VT>(0, 0, 1);
    checkEwBinarySca<BinaryOpCode::EQ, VT>(3, 3, 1);
    checkEwBinarySca<BinaryOpCode::EQ, VT>(3, 5, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::EQ>(VT("abcd"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("abce"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("abcda"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("abc"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("ABCD"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("36abcd"), VT("30abcd"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT("3"), VT("4"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT(""), VT("abc"), 0);
    checkEwBinarySca<BinaryOpCode::EQ>(VT(""), VT(""), 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::NEQ, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::NEQ, VT>(3, 3, 0);
    checkEwBinarySca<BinaryOpCode::NEQ, VT>(3, 5, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("abcd"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("abce"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("abcda"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("abc"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("ABCD"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("36abcd"), VT("30abcd"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT("3"), VT("4"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT(""), VT("abc"), 1);
    checkEwBinarySca<BinaryOpCode::NEQ>(VT(""), VT(""), 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LT, VT>(1, 1, 0);
    checkEwBinarySca<BinaryOpCode::LT, VT>(1, 3, 1);
    checkEwBinarySca<BinaryOpCode::LT, VT>(4, 2, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LT>(VT("abcd"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::LT>(VT("abce"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::LT>(VT("abcb"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT("abcda"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::LT>(VT("abc"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT("ABCD"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT("abcD"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT("36abcd"), VT("30abcd"), 0);
    checkEwBinarySca<BinaryOpCode::LT>(VT("3"), VT("4"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT(""), VT("abc"), 1);
    checkEwBinarySca<BinaryOpCode::LT>(VT(""), VT(""), 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::LE, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::LE, VT>(1, 3, 1);
    checkEwBinarySca<BinaryOpCode::LE, VT>(4, 2, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GT, VT>(1, 1, 0);
    checkEwBinarySca<BinaryOpCode::GT, VT>(1, 3, 0);
    checkEwBinarySca<BinaryOpCode::GT, VT>(4, 2, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GT>(VT("abcd"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT("abce"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::GT>(VT("abcb"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT("abcda"), VT("abcd"), 1);
    checkEwBinarySca<BinaryOpCode::GT>(VT("abc"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT("ABCD"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT("abcD"), VT("abcd"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT("36abcd"), VT("30abcd"), 1);
    checkEwBinarySca<BinaryOpCode::GT>(VT("3"), VT("4"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT(""), VT("abc"), 0);
    checkEwBinarySca<BinaryOpCode::GT>(VT(""), VT(""), 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::GE, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::GE, VT>(1, 3, 0);
    checkEwBinarySca<BinaryOpCode::GE, VT>(4, 2, 1);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MIN, VT>(2, 2, 2);
    checkEwBinarySca<BinaryOpCode::MIN, VT>(2, 3, 2);
    checkEwBinarySca<BinaryOpCode::MIN, VT>(3, 0, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::MAX, VT>(2, 2, 2);
    checkEwBinarySca<BinaryOpCode::MAX, VT>(2, 3, 3);
    checkEwBinarySca<BinaryOpCode::MAX, VT>(3, 0, 3);
}

// ****************************************************************************
// Logical
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::AND, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(0, 1, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(1, 0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::AND, VT>(0, 2, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(2, 0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(2, 2, 1);
    checkEwBinarySca<BinaryOpCode::AND, VT>(0, -2, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(-2, 0, 0);
    checkEwBinarySca<BinaryOpCode::AND, VT>(-2, -2, 1);
}

TEMPLATE_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<BinaryOpCode::OR, VT>(0, 0, 0);
    checkEwBinarySca<BinaryOpCode::OR, VT>(0, 1, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(1, 0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(1, 1, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(0, 2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(2, 0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(2, 2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(0, -2, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(-2, 0, 1);
    checkEwBinarySca<BinaryOpCode::OR, VT>(-2, -2, 1);
}

// ****************************************************************************
// String op
// ****************************************************************************
TEMPLATE_TEST_CASE(TEST_NAME("concat"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
    using VT = TestType;
    checkEwBinarySca<VT>(VT("abcd"), VT("abcd"), std::string("abcdabcd"));
    checkEwBinarySca<VT>(VT("ABCD"), VT("abcd"), std::string("ABCDabcd"));
    checkEwBinarySca<VT>(VT("3"), VT("4"), std::string("34"));
    checkEwBinarySca<VT>(VT(""), VT("abc"), std::string("abc"));
    checkEwBinarySca<VT>(VT(""), VT(""), std::string(""));
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    CHECK_THROWS(ewBinarySca<VT, VT, VT>(static_cast<BinaryOpCode>(999), 0, 0, nullptr));
}