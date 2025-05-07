/*
 * Copyright 2025 The DAPHNE Consortium
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
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColCalcBinary.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "ColCalcBinary (" opName ")"
#define DATA_TYPES Column
#define NUM_VALUE_TYPES double, uint32_t, int8_t
#define STR_VALUE_TYPES std::string

template <class DTArg, class DTRes>
void checkColCalcBinaryAndDestroy(BinaryOpCode opCode, const DTArg *lhs, const DTArg *rhs, const DTRes *exp) {
    DTRes *res = nullptr;
    colCalcBinary(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(lhs, rhs, exp, res);
}

// ****************************************************************************
// Valid arguments
// ****************************************************************************

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(4), VT(6)});
        exp = genGivenVals<DT>({VT(1), VT(6), VT(9)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::ADD, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sub"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(1), VT(6), VT(9)});
        rhs = genGivenVals<DT>({VT(1), VT(2), VT(3)});
        exp = genGivenVals<DT>({VT(0), VT(4), VT(6)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::SUB, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(4), VT(6)});
        exp = genGivenVals<DT>({VT(0), VT(8), VT(18)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MUL, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(8), VT(18)});
        rhs = genGivenVals<DT>({VT(1), VT(2), VT(3)});
        exp = genGivenVals<DT>({VT(0), VT(4), VT(6)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::DIV, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("pow"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(2), VT(2), VT(2)});
        rhs = genGivenVals<DT>({VT(0), VT(1), VT(3)});
        exp = genGivenVals<DT>({VT(1), VT(2), VT(8)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::POW, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mod"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(8), VT(0), VT(2)});
        rhs = genGivenVals<DT>({VT(3), VT(5), VT(3)});
        exp = genGivenVals<DT>({VT(2), VT(0), VT(2)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MOD, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("log"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(1), VT(2), VT(8)});
        rhs = genGivenVals<DT>({VT(2), VT(2), VT(2)});
        exp = genGivenVals<DT>({VT(0), VT(1), VT(3)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::LOG, lhs, rhs, exp);
}

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(1), VT(0), VT(1), VT(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::EQ, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(1), VTRes(0), VTRes(1), VTRes(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::EQ, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(0), VT(1), VT(0), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::NEQ, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(0), VTRes(1), VTRes(0), VTRes(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::NEQ, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(0), VT(1), VT(0), VT(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::LT, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(0), VTRes(1), VTRes(0), VTRes(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::LT, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(1), VT(1), VT(1), VT(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::LE, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(1), VTRes(1), VTRes(1), VTRes(0)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::LE, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(0), VT(0), VT(0), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::GT, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(0), VTRes(0), VTRes(0), VTRes(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::GT, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(1), VT(0), VT(1), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::GE, lhs, rhs, exp);
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTRes = int64_t;
    using DTRes = typename DT::template WithValueType<VTRes>;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DTRes *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DTRes>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DTRes>({VTRes(1), VTRes(0), VTRes(1), VTRes(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::GE, lhs, rhs, exp);
}

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MIN, lhs, rhs, exp);
}
#if 0 // not supported yet
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str1")});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MIN, lhs, rhs, exp);
}
#endif

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(2), VT(3)});
        rhs = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(1)});
        exp = genGivenVals<DT>({VT(0), VT(2), VT(2), VT(3)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MAX, lhs, rhs, exp);
}
#if 0 // not supported yet
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT("str0"), VT("str1"), VT("str2"), VT("str3")});
        rhs = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str1")});
        exp = genGivenVals<DT>({VT("str0"), VT("str2"), VT("str2"), VT("str3")});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::MAX, lhs, rhs, exp);
}
#endif

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(0), VT(1), VT(3), VT(1)});
        rhs = genGivenVals<DT>({VT(0), VT(0), VT(1), VT(1), VT(0), VT(3)});
        exp = genGivenVals<DT>({VT(0), VT(0), VT(0), VT(1), VT(0), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::AND, lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, (DATA_TYPES), (NUM_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(0), VT(1), VT(0), VT(1), VT(3), VT(1)});
        rhs = genGivenVals<DT>({VT(0), VT(0), VT(1), VT(1), VT(0), VT(3)});
        exp = genGivenVals<DT>({VT(0), VT(1), VT(1), VT(1), VT(1), VT(1)});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::OR, lhs, rhs, exp);
}

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("concat"), TAG_KERNELS, (DATA_TYPES), (STR_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *lhs = nullptr;
    DT *rhs = nullptr;
    DT *exp = nullptr;

    SECTION("empty inputs") {
        lhs = DataObjectFactory::create<DT>(0, false);
        rhs = DataObjectFactory::create<DT>(0, false);
        exp = DataObjectFactory::create<DT>(0, false);
    }
    SECTION("non-empty inputs") {
        lhs = genGivenVals<DT>({VT(""), VT("abc"), VT(""), VT("abc")});
        rhs = genGivenVals<DT>({VT(""), VT(""), VT("de"), VT("de")});
        exp = genGivenVals<DT>({VT(""), VT("abc"), VT("de"), VT("abcde")});
    }

    checkColCalcBinaryAndDestroy(BinaryOpCode::CONCAT, lhs, rhs, exp);
}

// ****************************************************************************
// Invalid arguments
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>({1});

    DT *res = nullptr;
    CHECK_THROWS(colCalcBinary(static_cast<BinaryOpCode>(999), res, arg, arg, nullptr));

    DataObjectFactory::destroy(arg);
    if (res)
        DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("any") ": size mismatch", TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;

    auto lhs = genGivenVals<DT>({1, 2, 3});
    auto rhs = genGivenVals<DT>({4, 5});

    DT *res = nullptr;
    CHECK_THROWS(colCalcBinary(BinaryOpCode::ADD, res, lhs, rhs, nullptr));

    DataObjectFactory::destroy(lhs, rhs);
    if (res)
        DataObjectFactory::destroy(res);
}