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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/EwBinaryMatSca.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

#define TEST_NAME(opName) "EwBinaryMatSca (" opName ")"
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkEwBinaryMatSca(BinaryOpCode opCode, const DT * lhs, typename DT::VT rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryMatSca<DT, DT, typename DT::VT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
}

// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            2, 3, 1, 1, 2, 4,
            1, 2, 1, 3, 1, 4,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
    });
    
    checkEwBinaryMatSca(BinaryOpCode::ADD, m0, 0, m0);
    checkEwBinaryMatSca(BinaryOpCode::ADD, m1, 0, m1);
    checkEwBinaryMatSca(BinaryOpCode::ADD, m1, 1, m2);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            2, 4, 0, 0, 2, 6,
            0, 2, 0, 4, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    
    checkEwBinaryMatSca(BinaryOpCode::MUL, m0, 0, m0);
    checkEwBinaryMatSca(BinaryOpCode::MUL, m1, 0, m0);
    checkEwBinaryMatSca(BinaryOpCode::MUL, m1, 2, m2);
        
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m0 = genGivenVals<DT>(2, {
            0, 0, 0,
            0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(2, {
            1, 2, 4,
            6, 8, 9,
    });
    auto m2 = genGivenVals<DT>(2, {
             2,  4,  8,
            12, 16, 18,
    });
    
    checkEwBinaryMatSca(BinaryOpCode::DIV, m0, 1, m0);
    checkEwBinaryMatSca(BinaryOpCode::DIV, m1, 1, m1);
    checkEwBinaryMatSca(BinaryOpCode::DIV, m2, 2, m1);
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 1, 0,  1, 0, 0,});
    
    checkEwBinaryMatSca(BinaryOpCode::EQ, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 0, 1,  0, 1, 1,});
    
    checkEwBinaryMatSca(BinaryOpCode::NEQ, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 0, 0,  0, 0, 1,});
    
    checkEwBinaryMatSca(BinaryOpCode::LT, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 1, 0,  1, 0, 1,});
    
    checkEwBinaryMatSca(BinaryOpCode::LE, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 0, 1,  0, 1, 0,});
    
    checkEwBinaryMatSca(BinaryOpCode::GT, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 1, 1,  1, 1, 0,});
    
    checkEwBinaryMatSca(BinaryOpCode::GE, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 2, 2,  2, 2, 1,});
    
    checkEwBinaryMatSca(BinaryOpCode::MIN, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {2, 2, 3,  2, 3, 2,});
    
    checkEwBinaryMatSca(BinaryOpCode::MAX, m1, 2, m2);
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * res = nullptr;
    auto m = genGivenVals<DT>(1, {1});
    CHECK_THROWS(ewBinaryMatSca<DT, DT, typename DT::VT>(static_cast<BinaryOpCode>(999), res, m, 1, nullptr));
}