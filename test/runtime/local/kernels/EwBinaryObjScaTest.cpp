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
#include <runtime/local/kernels/CastObj.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/EwBinaryObjSca.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

// TODO Currently, we only pass DenseMatrix as the data type to the template
// test cases. The Frame test cases are hard-coded on the template test cases.
// Once we add CSRMatrix here, we should also factor out the frame test cases.

#define TEST_NAME(opName) "EwBinaryObjSca (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, uint32_t

template<class DT, typename VT>
void checkEwBinaryObjSca(BinaryOpCode opCode, const DT * lhs, const VT rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryObjSca<DT, DT, VT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

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

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::ADD, m0, 0, m0);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::ADD, m1, 0, m1);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::ADD, m1, 1, m2);

    DataObjectFactory::destroy(m0, m1, m2);
}

TEMPLATE_TEST_CASE(TEST_NAME("add - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m0 = genGivenVals<DTCol>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DTCol>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DTCol>(4, {
            2, 3, 1, 1, 2, 4,
            1, 2, 1, 3, 1, 4,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
    });

    Frame * f0 = nullptr;
    castObj<Frame, DTCol>(f0, m0, nullptr);
    Frame * f1 = nullptr;
    castObj<Frame, DTCol>(f1, m1, nullptr);
    Frame * f2 = nullptr;
    castObj<Frame, DTCol>(f2, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::ADD, f0, 0, f0);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::ADD, f1, 0, f1);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::ADD, f1, 1, f2);

    DataObjectFactory::destroy(f0, f1, f2, m0, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

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

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::MUL, m0, 0, m0);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::MUL, m1, 0, m0);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::MUL, m1, 2, m2);

    DataObjectFactory::destroy(m0, m1, m2);
}

TEMPLATE_TEST_CASE(TEST_NAME("mul - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m0 = genGivenVals<DTCol>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DTCol>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DTCol>(4, {
            2, 4, 0, 0, 2, 6,
            0, 2, 0, 4, 0, 6,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });

    Frame * f0 = nullptr;
    castObj<Frame, DTCol>(f0, m0, nullptr);
    Frame * f1 = nullptr;
    castObj<Frame, DTCol>(f1, m1, nullptr);
    Frame * f2 = nullptr;
    castObj<Frame, DTCol>(f2, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::MUL, f0, 0, f0);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::MUL, f1, 0, f0);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::MUL, f1, 2, f2);

    DataObjectFactory::destroy(f0, f1, f2, m0, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto m0 = genGivenVals<DT>(2, {
            0, 0, 0,
            0, 0, 0
    });
    auto m1 = genGivenVals<DT>(2, {
            1, 2, 4,
            6, 8, 9
    });
    auto m2 = genGivenVals<DT>(2, {
            2,  4,  8,
            12, 16, 18
    });

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::DIV, m0, 1, m0);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::DIV, m1, 1, m1);
    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::DIV, m2, 2, m1);

    DataObjectFactory::destroy(m0, m1, m2);
}

TEMPLATE_TEST_CASE(TEST_NAME("div - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m0 = genGivenVals<DTCol>(2, {
            0, 0, 0,
            0, 0, 0
    });
    auto m1 = genGivenVals<DTCol>(2, {
            1, 2, 4,
            6, 8, 9
    });
    auto m2 = genGivenVals<DTCol>(2, {
            2,  4,  8,
            12, 16, 18
    });

    Frame * f0 = nullptr;
    castObj<Frame, DTCol>(f0, m0, nullptr);
    Frame * f1 = nullptr;
    castObj<Frame, DTCol>(f1, m1, nullptr);
    Frame * f2 = nullptr;
    castObj<Frame, DTCol>(f2, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::DIV, f0, 1, f0);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::DIV, f1, 1, f1);
    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::DIV, f2, 2, f1);

    DataObjectFactory::destroy(f0, f1, f2, m0, m1, m2);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {0, 1, 0, 1, 0, 0});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::EQ, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("eq - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {0, 1, 0, 1, 0, 0,});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::EQ, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {1, 0, 1, 0, 1, 1});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::NEQ, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("neq - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {1, 0, 1, 0, 1, 1});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::NEQ, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {1, 0, 0, 0, 0, 1});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::LT, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("lt - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {1, 0, 0, 0, 0, 1});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::LT, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {1, 1, 0, 1, 0, 1});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::LE, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("le - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {1, 1, 0, 1, 0, 1});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::LE, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {0, 0, 1, 0, 1, 0});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::GT, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("gt - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {0, 0, 1, 0, 1, 0});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::GT, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {0, 1, 1, 1, 1, 0});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::GE, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("ge - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {0, 1, 1, 1, 1, 0});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::GE, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {1, 2, 2, 2, 2, 1});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::MIN, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("min - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {1, 2, 2, 2, 2, 1});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::MIN, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {1, 2, 3, 2, 3, 1});
    auto exp = genGivenVals<DT>(2, {2, 2, 3, 2, 3, 2});

    checkEwBinaryObjSca<DT, VT>(BinaryOpCode::MAX, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("max - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {1, 2, 3, 2, 3, 1});
    auto m2 = genGivenVals<DTCol>(2, {2, 2, 3, 2, 3, 2});

    Frame * arg = nullptr;
    castObj<Frame, DTCol>(arg, m1, nullptr);
    Frame * exp = nullptr;
    castObj<Frame, DTCol>(exp, m2, nullptr);

    checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::MAX, arg, 2, exp);

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

// ****************************************************************************
// Logical
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("and - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {0, 1, 2, VT(-2)});
    DT * exp = nullptr;

    SECTION("scalar=0, matrix") {
        exp = genGivenVals<DT>(2, {0, 0, 0, 0});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::AND, arg, 0, exp);
    }
    SECTION("scalar=1, matrix") {
        exp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::AND, arg, 1, exp);
    }
    SECTION("scalar=2, matrix") {
        exp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::AND, arg, 2, exp);
    }
    SECTION("scalar=-2, matrix") {
        exp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::AND, arg, VT(-2), exp);
    }

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("and - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {0, 1, 2, VT(-2)});
    DTCol * m2 = nullptr;
    Frame * arg = nullptr;
    Frame * exp = nullptr;

    SECTION("scalar=0, frame") {
        m2 = genGivenVals<DTCol>(2, {0, 0, 0, 0});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::AND, arg, 0, exp);
    }
    SECTION("scalar=1, frame") {
        m2 = genGivenVals<DTCol>(2, {0, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::AND, arg, 1, exp);
    }
    SECTION("scalar=2, frame") {
        m2 = genGivenVals<DTCol>(2, {0, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::AND, arg, 2, exp);
    }
    SECTION("scalar=-2, frame") {
        m2 = genGivenVals<DTCol>(2, {0, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::AND, arg, VT(-2), exp);
    }

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("or - Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {0, 1, 2, VT(-2)});
    DT * exp = nullptr;

    SECTION("scalar=0, matrix") {
        exp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::OR, arg, 0, exp);
    }
    SECTION("scalar=1, matrix") {
        exp = genGivenVals<DT>(2, {1, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::OR, arg, 1, exp);
    }
    SECTION("scalar=2, matrix") {
        exp = genGivenVals<DT>(2, {1, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::OR, arg, 2, exp);
    }
    SECTION("scalar=-2, matrix") {
        exp = genGivenVals<DT>(2, {1, 1, 1, 1});
        checkEwBinaryObjSca<DT, VT>(BinaryOpCode::OR, arg, VT(-2), exp);
    }

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_TEST_CASE(TEST_NAME("or - Frame"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using DTCol = DenseMatrix<VT>;

    auto m1 = genGivenVals<DTCol>(2, {0, 1, 2, VT(-2)});
    DTCol * m2 = nullptr;
    Frame * arg = nullptr;
    Frame * exp = nullptr;

    SECTION("scalar=0, frame") {
        m2 = genGivenVals<DTCol>(2, {0, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::OR, arg, 0, exp);
    }
    SECTION("scalar=1, frame") {
        m2 = genGivenVals<DTCol>(2, {1, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::OR, arg, 1, exp);
    }
    SECTION("scalar=2, frame") {
        m2 = genGivenVals<DTCol>(2, {1, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::OR, arg, 2, exp);
    }
    SECTION("scalar=-2, frame") {
        m2 = genGivenVals<DTCol>(2, {1, 1, 1, 1});
        castObj<Frame, DTCol>(arg, m1, nullptr);
        castObj<Frame, DTCol>(exp, m2, nullptr);

        checkEwBinaryObjSca<Frame, VT>(BinaryOpCode::OR, arg, VT(-2), exp);
    }

    DataObjectFactory::destroy(arg, exp, m1, m2);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * res = nullptr;
    auto arg = genGivenVals<DT>(1, {1});
    CHECK_THROWS(ewBinaryObjSca<DT, DT, typename DT::VT>(static_cast<BinaryOpCode>(999), res, arg, 1, nullptr));
}