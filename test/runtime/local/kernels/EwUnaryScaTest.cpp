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

#include <runtime/local/kernels/EwUnarySca.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>

#include <cstdint>

#define TEST_NAME(opName) "EwUnarySca (" opName ")"
#define SI_VALUE_TYPES int8_t, int32_t, int64_t
#define FP_VALUE_TYPES float, double
#define VALUE_TYPES SI_VALUE_TYPES, FP_VALUE_TYPES

template<UnaryOpCode opCode, typename VT>
void checkEwUnarySca(const VT arg, const VT exp) {
    CHECK(EwUnarySca<opCode, VT, VT>::apply(arg, nullptr) == exp);
    CHECK(ewUnarySca<VT, VT>(opCode, arg, nullptr) == exp);
}

// decimals below are implicitely truncated if VALUE_TYPES is integer
// generally avoid this for input arguments
template<UnaryOpCode opCode, typename VT>
void checkEwUnaryScaApprox(const VT arg, const VT exp) {
    CHECK(Approx(EwUnarySca<opCode, VT, VT>::apply(arg, nullptr)).epsilon(1e-2) == exp);
    CHECK(Approx(ewUnarySca<VT, VT>(opCode, arg, nullptr)).epsilon(1e-2) == exp);
}

template<UnaryOpCode opCode, typename VT>
void checkEwUnaryScaThrow(const VT arg) {
    REQUIRE_THROWS_AS((EwUnarySca<opCode, VT, VT>::apply(arg, nullptr)), std::domain_error);
    REQUIRE_THROWS_AS((ewUnarySca<VT, VT>(opCode, arg, nullptr)), std::domain_error);
}

template<UnaryOpCode opCode, typename VT>
void checkEwUnaryScaNaN(const VT arg) {
    VT res1 = EwUnarySca<opCode, VT, VT>::apply(arg, nullptr);
    VT res2 = ewUnarySca<VT, VT>(opCode, arg, nullptr);
    CHECK(res1 != res1);
    CHECK(res2 != res2);
}

// ****************************************************************************
// Arithmetic/general math
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("abs"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ABS, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::ABS, VT>(2, 2);
    checkEwUnarySca<UnaryOpCode::ABS, VT>(-2, 2);
}

TEMPLATE_TEST_CASE(TEST_NAME("sign"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(1, 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(123, 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-1, -1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-123, -1);
}

TEMPLATE_TEST_CASE(TEST_NAME("sign, floating-point-specific"), TAG_KERNELS, FP_VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(0.3, 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-0.3, -1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(1.1, 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(123.4, 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-1.1, -1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-123.4, -1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(std::numeric_limits<VT>::infinity(), 1);
    checkEwUnarySca<UnaryOpCode::SIGN, VT>(-std::numeric_limits<VT>::infinity(), -1);
    checkEwUnaryScaNaN<UnaryOpCode::SIGN, VT>(std::numeric_limits<VT>::quiet_NaN());
}

TEMPLATE_TEST_CASE(TEST_NAME("sqrt"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::SQRT, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::SQRT, VT>(1, 1);
    checkEwUnarySca<UnaryOpCode::SQRT, VT>(16, 4);
    checkEwUnaryScaThrow<UnaryOpCode::SQRT, VT>(-1);
}

TEMPLATE_TEST_CASE(TEST_NAME("exp"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::EXP, VT>(0, 1.0);
    checkEwUnaryScaApprox<UnaryOpCode::EXP, VT>(-1, 0.367);
    checkEwUnaryScaApprox<UnaryOpCode::EXP, VT>(3, 20.085);
}

TEMPLATE_TEST_CASE(TEST_NAME("ln"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::LN, VT>(1, 0);
    checkEwUnaryScaApprox<UnaryOpCode::LN, VT>(3, 1.098);
    checkEwUnaryScaApprox<UnaryOpCode::LN, VT>(8, 2.079);
    checkEwUnaryScaThrow<UnaryOpCode::LN, VT>(-1);
}

TEMPLATE_TEST_CASE(TEST_NAME("ln, floating-point-specific"), TAG_KERNELS, FP_VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::LN, VT>(0, -std::numeric_limits<VT>::infinity());
}

// ****************************************************************************
// Trigonometric/Hyperbolic functions
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("sin"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::SIN, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::SIN, VT>(1, 0.841);
    checkEwUnaryScaApprox<UnaryOpCode::SIN, VT>(-1, -0.841);
}

TEMPLATE_TEST_CASE(TEST_NAME("cos"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::COS, VT>(0, 1);
    checkEwUnaryScaApprox<UnaryOpCode::COS, VT>(1, 0.54);
    checkEwUnaryScaApprox<UnaryOpCode::COS, VT>(-1, 0.54);
}

TEMPLATE_TEST_CASE(TEST_NAME("tan"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::TAN, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::TAN, VT>(1, 1.557);
    checkEwUnaryScaApprox<UnaryOpCode::TAN, VT>(-1, -1.557);
}

TEMPLATE_TEST_CASE(TEST_NAME("asin"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ASIN, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::ASIN, VT>(1, 1.57);
    checkEwUnaryScaApprox<UnaryOpCode::ASIN, VT>(-1, -1.57);
    checkEwUnaryScaThrow<UnaryOpCode::ASIN, VT>(-2);
    checkEwUnaryScaThrow<UnaryOpCode::ASIN, VT>(2);
}

TEMPLATE_TEST_CASE(TEST_NAME("acos"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ACOS, VT>(1, 0);
    checkEwUnaryScaApprox<UnaryOpCode::ACOS, VT>(0, 1.57);
    checkEwUnaryScaApprox<UnaryOpCode::ACOS, VT>(-1, 3.141);
    checkEwUnaryScaThrow<UnaryOpCode::ACOS, VT>(-2);
    checkEwUnaryScaThrow<UnaryOpCode::ACOS, VT>(2);
}

TEMPLATE_TEST_CASE(TEST_NAME("atan"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ATAN, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::ATAN, VT>(1, 0.785);
    checkEwUnaryScaApprox<UnaryOpCode::ATAN, VT>(-1, -0.785);
}

TEMPLATE_TEST_CASE(TEST_NAME("sinh"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::SINH, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::SINH, VT>(1, 1.175);
    checkEwUnaryScaApprox<UnaryOpCode::SINH, VT>(-1, -1.175);
}

TEMPLATE_TEST_CASE(TEST_NAME("cosh"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::COSH, VT>(0, 1);
    checkEwUnaryScaApprox<UnaryOpCode::COSH, VT>(1, 1.543);
    checkEwUnaryScaApprox<UnaryOpCode::COSH, VT>(-1, 1.543);
}

TEMPLATE_TEST_CASE(TEST_NAME("tanh"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::TANH, VT>(0, 0);
    checkEwUnaryScaApprox<UnaryOpCode::TANH, VT>(1, 0.761);
    checkEwUnaryScaApprox<UnaryOpCode::TANH, VT>(-1, -0.761);
}

// ****************************************************************************
// Rounding
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("floor"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(1, 1);
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(-1, -1);
}

TEMPLATE_TEST_CASE(TEST_NAME("floor, floating-point-specific"), TAG_KERNELS, FP_VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(0.3, 0);
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(-0.3, -1);
    checkEwUnarySca<UnaryOpCode::FLOOR, VT>(0.9, 0);
}

TEMPLATE_TEST_CASE(TEST_NAME("ceil"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(1, 1);
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(-1, -1);
}

TEMPLATE_TEST_CASE(TEST_NAME("ceil, floating-point-specific"), TAG_KERNELS, FP_VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(0.3, 1);
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(-0.3, 0);
    checkEwUnarySca<UnaryOpCode::CEIL, VT>(-1.1, -1);
}

TEMPLATE_TEST_CASE(TEST_NAME("round"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(0, 0);
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(1, 1);
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(-1, -1);
}

TEMPLATE_TEST_CASE(TEST_NAME("round, floating-point-specific"), TAG_KERNELS, FP_VALUE_TYPES) {
    using VT = TestType;
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(0.3, 0);
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(-0.3, 0);
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(0.5, 1);
    checkEwUnarySca<UnaryOpCode::ROUND, VT>(-0.5, -1);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    CHECK_THROWS(ewUnarySca<VT, VT>(static_cast<UnaryOpCode>(999), 0, nullptr));
}