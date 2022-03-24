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
#define DATA_TYPES DenseMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkEwBinaryMatSca(BinaryOpCode opCode, const DT * lhs, typename DT::VT rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryObjSca<DT, DT, typename DT::VT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template<class DT, typename VT>
void checkEwBinaryFrameSca(BinaryOpCode opCode, const DT * lhs, VT rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryObjSca<DT, DT, VT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
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

    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::ADD, m0, 0, m0);
        checkEwBinaryMatSca(BinaryOpCode::ADD, m1, 0, m1);
        checkEwBinaryMatSca(BinaryOpCode::ADD, m1, 1, m2);
    }   
    SECTION("frame") {
        Frame * f0 = nullptr;
        castObj<Frame, DT>(f0, m0, nullptr);
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::ADD, f0, 0, f0);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::ADD, f1, 0, f1);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::ADD, f1, 1, f2);
        DataObjectFactory::destroy(f0, f1, f2);
    }

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
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
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::MUL, m0, 0, m0);
        checkEwBinaryMatSca(BinaryOpCode::MUL, m1, 0, m0);
        checkEwBinaryMatSca(BinaryOpCode::MUL, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f0 = nullptr;
        castObj<Frame, DT>(f0, m0, nullptr);
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::MUL, f0, 0, f0);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::MUL, f1, 0, f0);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::MUL, f1, 2, f2);
        DataObjectFactory::destroy(f0, f1, f2);
    }
    
        
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
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
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::DIV, m0, 1, m0);
        checkEwBinaryMatSca(BinaryOpCode::DIV, m1, 1, m1);
        checkEwBinaryMatSca(BinaryOpCode::DIV, m2, 2, m1);
    }   
    SECTION("frame") {
        Frame * f0 = nullptr;
        castObj<Frame, DT>(f0, m0, nullptr);
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::DIV, f0, 1, f0);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::DIV, f1, 1, f1);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::DIV, f2, 2, f1);
        DataObjectFactory::destroy(f0, f1, f2);
    }
    
    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 1, 0,  1, 0, 0,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::EQ, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::EQ, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 0, 1,  0, 1, 1,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::NEQ, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::NEQ, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 0, 0,  0, 0, 1,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::LT, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::LT, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 1, 0,  1, 0, 1,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::LE, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::LE, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
        
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 0, 1,  0, 1, 0,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::GT, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::GT, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
        
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {0, 1, 1,  1, 1, 0,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::GE, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::GE, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }
        
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {1, 2, 2,  2, 2, 1,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::MIN, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::MIN, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  2, 3, 1});
    auto m2 = genGivenVals<DT>(2, {2, 2, 3,  2, 3, 2,});
    
    SECTION("matrix") {
        checkEwBinaryMatSca(BinaryOpCode::MAX, m1, 2, m2);
    }   
    SECTION("frame") {
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * f2 = nullptr;
        castObj<Frame, DT>(f2, m2, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::MAX, f1, 2, f2);
        DataObjectFactory::destroy(f1, f2);
    }    

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// Logical
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {0, 1, 2, VT(-2)});
    
    DT * mExp = nullptr;
    SECTION("scalar=0, matrix") {
        mExp = genGivenVals<DT>(2, {0, 0, 0, 0});
        checkEwBinaryMatSca(BinaryOpCode::AND, m1, 0, mExp);
    }
    SECTION("scalar=1, matrix") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryMatSca(BinaryOpCode::AND, m1, 1, mExp);
    }
    SECTION("scalar=2, matrix") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryMatSca(BinaryOpCode::AND, m1, 2, mExp);
    }
    SECTION("scalar=-2, matrix") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        checkEwBinaryMatSca(BinaryOpCode::AND, m1, VT(-2), mExp);
    }
    SECTION("scalar=0, frame") {
        mExp = genGivenVals<DT>(2, {0, 0, 0, 0});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::AND, f1, 0, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=1, frame") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::AND, f1, 1, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=2, frame") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::AND, f1, 2, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=-2, frame") {
        mExp = genGivenVals<DT>(2, {0, 1, 1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::AND, f1, VT(-2), fExp);
        DataObjectFactory::destroy(f1, fExp);
    }

    DataObjectFactory::destroy(m1, mExp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    
    auto m1 = genGivenVals<DT>(2, {0, 1,  2, VT(-2)});
    
    DT * mExp = nullptr;
    SECTION("scalar=0, matrix") {
        mExp = genGivenVals<DT>(2, {0, 1,  1, 1});
        checkEwBinaryMatSca(BinaryOpCode::OR, m1, 0, mExp);
    }
    SECTION("scalar=1, matrix") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        checkEwBinaryMatSca(BinaryOpCode::OR, m1, 1, mExp);
    }
    SECTION("scalar=2, matrix") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        checkEwBinaryMatSca(BinaryOpCode::OR, m1, 2, mExp);
    }
    SECTION("scalar=-2, matrix") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        checkEwBinaryMatSca(BinaryOpCode::OR, m1, VT(-2), mExp);
    }
    SECTION("scalar=0, frame") {
        mExp = genGivenVals<DT>(2, {0, 1,  1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::OR, f1, 0, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=1, frame") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::OR, f1, 1, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=2, frame") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::OR, f1, 2, fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    SECTION("scalar=-2, frame") {
        mExp = genGivenVals<DT>(2, {1, 1,  1, 1});
        Frame * f1 = nullptr;
        castObj<Frame, DT>(f1, m1, nullptr);
        Frame * fExp = nullptr;
        castObj<Frame, DT>(fExp, mExp, nullptr);
        checkEwBinaryFrameSca<Frame,VT>(BinaryOpCode::OR, f1, VT(-2), fExp);
        DataObjectFactory::destroy(f1, fExp);
    }
    
    DataObjectFactory::destroy(m1, mExp);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * res = nullptr;
    auto m = genGivenVals<DT>(1, {1});
    CHECK_THROWS(ewBinaryObjSca<DT, DT, typename DT::VT>(static_cast<BinaryOpCode>(999), res, m, 1, nullptr));
}