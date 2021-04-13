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

template<BinaryOpCode opCode, typename VT>
void checkEwBinarySca(VT lhs, VT rhs, VT exp) {
    CHECK(ewBinaryScaCT<opCode, VT, VT, VT>(lhs, rhs) == exp);
    CHECK(ewBinaryScaRT<VT, VT, VT>(opCode, lhs, rhs) == exp);
}

TEMPLATE_TEST_CASE("EwBinarySca", TAG_KERNELS, double, uint32_t) {
    using VT = TestType;

    SECTION("add") {
        checkEwBinarySca<BinaryOpCode::ADD, VT>(0, 0, 0);
        checkEwBinarySca<BinaryOpCode::ADD, VT>(0, 1, 1);
        checkEwBinarySca<BinaryOpCode::ADD, VT>(1, 2, 3);
    }
    SECTION("mul") {
        checkEwBinarySca<BinaryOpCode::MUL, VT>(0, 0, 0);
        checkEwBinarySca<BinaryOpCode::MUL, VT>(0, 1, 0);
        checkEwBinarySca<BinaryOpCode::MUL, VT>(2, 3, 6);
    }
    SECTION("some invalid opCode") {
        CHECK_THROWS(ewBinaryScaRT<VT, VT, VT>(static_cast<BinaryOpCode>(999), 0, 0));
    }
}