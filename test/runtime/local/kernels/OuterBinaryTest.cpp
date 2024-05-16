/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/OuterBinary.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

#include <cstdint>

#define TEST_NAME(opName) "OuterBinary (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, int32_t

// ****************************************************************************
// Helpers
// ****************************************************************************

template<class DT>
void checkOuterBinary(BinaryOpCode opCode, const DT * lhs, const DT * rhs, const DT * exp) {
    DT * res = nullptr;
    outerBinary<DT, DT, DT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
}

template<class DT>
void helper(BinaryOpCode opCode, std::vector<typename DT::VT> lhsVals, std::vector<typename DT::VT> rhsVals, std::vector<typename DT::VT> resVals) {
    if(resVals.size() != lhsVals.size() * rhsVals.size())
        throw std::runtime_error(
                "the number of elements in resVals must be the product "
                "of the numbers of elements in lhsVals and rhsVals"
        );

    auto lhs = genGivenVals<DT>(lhsVals.size(), lhsVals);
    auto rhs = genGivenVals<DT>(1, rhsVals);
    auto exp = genGivenVals<DT>(resVals.size(), resVals);
    
    checkOuterBinary(opCode, lhs, rhs, exp);

    DataObjectFactory::destroy(lhs, rhs, exp);
}

// ****************************************************************************
// Tests for various input/output shapes
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("valid shapes"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTEmpty = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;

    DT * lhs = nullptr;
    DT * rhs = nullptr;
    DT * exp = nullptr;

    SECTION("0x1 op 1x0") {
        lhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 1, false));
        rhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(1, 0, false));
        exp = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
    }
    SECTION("0x1 op 1xn") {
        lhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 1, false));
        rhs = genGivenVals<DT>(1, {4, 5, 6, 7});
        exp = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 4, false));
    }
    SECTION("mx1 op 1x0") {
        lhs = genGivenVals<DT>(3, {1, 2, 3});
        rhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(1, 0, false));
        exp = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(3, 0, false));
    }
    SECTION("mx1 op 1xn") {
        lhs = genGivenVals<DT>(3, {1, 2, 3});
        rhs = genGivenVals<DT>(1, {4, 5, 6, 7});
        exp = genGivenVals<DT>(3, {
            5, 6, 7,  8,
            6, 7, 8,  9,
            7, 8, 9, 10,
        });
    }
    
    checkOuterBinary(BinaryOpCode::ADD, lhs, rhs, exp);

    DataObjectFactory::destroy(lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("invalid shapes"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTEmpty = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;

    DT * lhs = nullptr;
    DT * rhs = nullptr;

    SECTION("mx0 op 1xn") {
        lhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(3, 0, false));
        rhs = genGivenVals<DT>(1, {0, 0, 0, 0});
    }
    SECTION("mx1 op 0xn") {
        lhs = genGivenVals<DT>(3, {0, 0, 0});
        rhs = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 4, false));
    }
    SECTION("mx2 op 1xn") {
        lhs = genGivenVals<DT>(3, {0, 0,  0, 0,  0, 0});
        rhs = genGivenVals<DT>(1, {0, 0, 0, 0});
    }
    SECTION("mx1 op 2xn") {
        lhs = genGivenVals<DT>(3, {0, 0, 0});
        rhs = genGivenVals<DT>(2, {0, 0, 0, 0,  0, 0, 0, 0});
    }
    
    DT * res = nullptr;
    CHECK_THROWS(outerBinary<DT, DT, DT>(BinaryOpCode::ADD, res, lhs, rhs, nullptr));
    DataObjectFactory::destroy(lhs, rhs);
    if(res)
        DataObjectFactory::destroy(res);
}

// ****************************************************************************
// Tests for various op codes
// ****************************************************************************

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::ADD, {1, 2}, {3}, {4, 5});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sub"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::SUB, {1, 2}, {3}, {-2, -1});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::MUL, {1, 2}, {3}, {3, 6});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::DIV, {6, 4}, {2}, {3, 2});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("pow"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::POW, {1, 2}, {3}, {1, 8});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mod"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::MOD, {7, 2}, {3}, {1, 2});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("log"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::LOG, {16, 8}, {2}, {4, 3});
}

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::MIN, {1, 3}, {2}, {1, 2});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::MAX, {1, 3}, {2}, {2, 3});
}

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::AND, {1, 0}, {1}, {1, 0});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::OR, {1, 0}, {1}, {1, 1});
}

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::EQ, {1, 2}, {2}, {0, 1});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::NEQ, {1, 2}, {2}, {1, 0});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::LT, {1, 2}, {2}, {1, 0});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::LE, {1, 2}, {2}, {1, 1});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::GT, {1, 2}, {2}, {0, 0});
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    helper<TestType>(BinaryOpCode::GE, {1, 2}, {2}, {0, 1});
}

// ----------------------------------------------------------------------------
// Invalid op-code
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * res = nullptr;
    auto m = genGivenVals<DT>(1, {1});
    CHECK_THROWS(outerBinary<DT, DT, DT>(static_cast<BinaryOpCode>(999), res, m, m, nullptr));
}