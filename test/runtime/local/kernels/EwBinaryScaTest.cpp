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

template<BinaryOpCode opCode, typename VT>
void checkEwBinarySca(VT lhs, VT rhs, VT exp) {
    CHECK(ewBinaryScaCT<opCode, VT, VT, VT>(lhs, rhs) == exp);
    CHECK(ewBinaryScaRT<VT, VT, VT>(opCode, lhs, rhs) == exp);
}

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

TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    CHECK_THROWS(ewBinaryScaRT<VT, VT, VT>(static_cast<BinaryOpCode>(999), 0, 0));
}