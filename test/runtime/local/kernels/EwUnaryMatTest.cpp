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

#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>

#include <cstdint>

#define TEST_NAME(opName) "EwUnaryMat (" opName ")"
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES int32_t, double

template<typename DTRes, typename DTArg>
void checkEwUnaryMat(UnaryOpCode opCode, const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    ewUnaryMat<DTRes, DTArg>(opCode, res, arg, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template<typename DTRes, typename DTArg>
void checkEwUnaryMatApprox(UnaryOpCode opCode, const DTArg * arg, const DTRes * exp) {
    DTRes * res = nullptr;
    ewUnaryMat<DTRes, DTArg>(opCode, res, arg, nullptr);
    CHECK(checkEqApprox(res, exp, 1e-2, nullptr));
    DataObjectFactory::destroy(res);
}

template<typename DTArg>
void checkEwUnaryMatThrow(UnaryOpCode opCode, const DTArg * arg) {
    DTArg * res = nullptr;
    REQUIRE_THROWS_AS((ewUnaryMat<DTArg, DTArg>(opCode, res, arg, nullptr)), std::domain_error);
    DataObjectFactory::destroy(res);
}

// ****************************************************************************
// Arithmetic/general math
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        1,
        1,
    });

    checkEwUnaryMat(UnaryOpCode::ABS, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sign"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {
        0, 1, -1,
        10, -10, VT(1.4),
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(2, {
        0, 1, -1,
        1, -1, 1,
    });

    checkEwUnaryMat(UnaryOpCode::SIGN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sign, floating-point-specific"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {
        std::numeric_limits<VT>::infinity(),
        - std::numeric_limits<VT>::infinity(),
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(2, {
        1,
        -1,
    });

    checkEwUnaryMat(UnaryOpCode::SIGN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sqrt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        16,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        1,
        4,
    });

    checkEwUnaryMat(UnaryOpCode::SQRT, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sqrt, check domain_error"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    checkEwUnaryMatThrow(UnaryOpCode::SQRT, arg);

    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("exp"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        -1,
        3,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        1,
        VT(0.367),
        VT(20.085),
    });

    checkEwUnaryMatApprox(UnaryOpCode::EXP, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ln"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        1,
        3,
        8,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(1.098),
        VT(2.079),
    });

    checkEwUnaryMatApprox(UnaryOpCode::LN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ln, check domain_error"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    checkEwUnaryMatThrow(UnaryOpCode::LN, arg);

    DataObjectFactory::destroy(arg);
}

// ****************************************************************************
// Trigonometric/Hyperbolic functions
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sin"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(0.841),
        VT(-0.841),
    });

    checkEwUnaryMatApprox(UnaryOpCode::SIN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("cos"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        1,
        VT(0.54),
        VT(0.54),
    });

    checkEwUnaryMatApprox(UnaryOpCode::COS, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("tan"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(1.557),
        VT(-1.557),
    });

    checkEwUnaryMatApprox(UnaryOpCode::TAN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("asin"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(1.57),
        VT(-1.57),
    });

    checkEwUnaryMatApprox(UnaryOpCode::ASIN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("asin, check domain_error"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -2,
    });

    checkEwUnaryMatThrow(UnaryOpCode::ASIN, arg);

    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("acos"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        VT(1.57),
        0,
        VT(3.141),
    });

    checkEwUnaryMatApprox(UnaryOpCode::ACOS, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("acos, check domain_error"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -2,
    });

    checkEwUnaryMatThrow(UnaryOpCode::ACOS, arg);

    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("atan"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(0.785),
        VT(-0.785),
    });

    checkEwUnaryMatApprox(UnaryOpCode::ATAN, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sinh"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(1.175),
        VT(-1.175),
    });

    checkEwUnaryMatApprox(UnaryOpCode::SINH, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("cosh"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        1,
        VT(1.543),
        VT(1.543),
    });

    checkEwUnaryMatApprox(UnaryOpCode::COSH, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("tanh"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        VT(0.761),
        VT(-0.761),
    });

    checkEwUnaryMatApprox(UnaryOpCode::TANH, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

// ****************************************************************************
// Rounding
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("floor"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        1,
        -1,
    });

    checkEwUnaryMat(UnaryOpCode::FLOOR, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("floor, floating-point-specific"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {
        0.3, -0.3,
        0.9, -0.9,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(2, {
        0, -1,
        0, -1,
    });

    checkEwUnaryMat(UnaryOpCode::FLOOR, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ceil"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        1,
        -1,
    });

    checkEwUnaryMat(UnaryOpCode::CEIL, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ceil, floating-point-specific"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {
        0.3, -0.3,
        1.1, -1.9,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(2, {
        1, -0.0,
        2, -1,
    });

    checkEwUnaryMat(UnaryOpCode::CEIL, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("round"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
        0,
        1,
        -1,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(3, {
        0,
        1,
        -1,
    });

    checkEwUnaryMat(UnaryOpCode::ROUND, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("round, floating-point-specific"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(2, {
        0.3, -0.3,
        0.5, -0.5,
    });

    auto dense_exp = genGivenVals<DenseMatrix<VT>>(2, {
        0, -0.0,
        1, -1,
    });

    checkEwUnaryMat(UnaryOpCode::ROUND, arg, dense_exp);

    DataObjectFactory::destroy(arg, dense_exp);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    auto arg = genGivenVals<DT>(1, {1});
    DT * exp = nullptr;
    CHECK_THROWS(ewUnaryMat<DT, DT>(static_cast<UnaryOpCode>(999), exp, arg, nullptr));

    DataObjectFactory::destroy(arg);
}