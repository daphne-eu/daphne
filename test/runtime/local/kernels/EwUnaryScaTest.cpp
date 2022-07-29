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
void checkEwUnarySca(VT arg, VT exp) {
    if constexpr(std::is_same_v<VT, StringScalarType>){
        CHECK(strcmp(EwUnarySca<opCode, VT, VT>::apply(arg, nullptr), exp) == 0);
        CHECK(strcmp(ewUnarySca<VT, VT>(opCode, arg, nullptr), exp) == 0);
    }else{
        CHECK(EwUnarySca<opCode, VT, VT>::apply(arg, nullptr) == exp);
        CHECK(ewUnarySca<VT, VT>(opCode, arg, nullptr) == exp);
    }
}

template<UnaryOpCode opCode, typename VT>
void checkEwUnaryScaNaN(VT arg) {
    VT res1 = EwUnarySca<opCode, VT, VT>::apply(arg, nullptr);
    VT res2 = ewUnarySca<VT, VT>(opCode, arg, nullptr);
    CHECK(res1 != res1);
    CHECK(res2 != res2);
}

// ****************************************************************************
// Arithmetic/general math
// ****************************************************************************

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

// ****************************************************************************
// String
// ****************************************************************************
TEMPLATE_TEST_CASE(TEST_NAME("Lower/UpperCase"), TAG_KERNELS, VALUE_TYPES) {
    using VT = const char* ;
    VT arg = "hElLo";
    VT expUpper = "HELLO";
    VT expLower = "hello";
    checkEwUnarySca<UnaryOpCode::UPPERCASE, VT>(arg, expUpper);
    checkEwUnarySca<UnaryOpCode::LOWERCASE, VT>(arg, expLower);
}


// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    CHECK_THROWS(ewUnarySca<VT, VT>(static_cast<UnaryOpCode>(999), 0, nullptr));
}