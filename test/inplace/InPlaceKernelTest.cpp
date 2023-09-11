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

#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwBinaryObjSca.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/Transpose.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>
#include <cstdint>

// ****************************************************************************
// ewBinaryMat
// ****************************************************************************

template<class DT>
void checkEwBinaryMat(BinaryOpCode opCode, DT * lhs, DT * rhs, const DT * exp, bool hasFutureUseLhs, bool hasFutureUseRhs) {
    DT * res = nullptr;
    ewBinaryMat<DT, DT, DT>(opCode, res, lhs, rhs, hasFutureUseLhs, hasFutureUseRhs, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("ewBinaryMat - In-Place", TAG_INPLACE, (DenseMatrix), (uint32_t)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            1, 2, 3, 1, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 1, 0, 2,
    });
    auto m3 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            1, 3, 3, 3, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 1, 0, 2,
    });
    
    auto variations = {std::make_tuple(false, true),
                       std::make_tuple(true, false),
                       std::make_tuple(false, false)};

    for(auto var : variations) {
        DYNAMIC_SECTION("InPlaceOperand: lhs(" << std::get<0>(var) << ") rhs(" << std::get<1>(var) << ")") {
            checkEwBinaryMat(BinaryOpCode::ADD, m1, m2, m3, std::get<0>(var), std::get<1>(var));
        }
    }
    
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

// ****************************************************************************
// ewBinaryObjSca
// ****************************************************************************

template<class DT>
void checkEwBinaryMatSca(BinaryOpCode opCode, DT * lhs, typename DT::VT rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryObjSca<DT, DT, typename DT::VT>(opCode, res, lhs, rhs, false, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ewBinaryObjSca - In-Place", TAG_INPLACE, (DenseMatrix), (uint32_t)) {
    using DT = TestType;
    //using VT = typename DT::VT;
    
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

    checkEwBinaryMatSca(BinaryOpCode::ADD, m1, 1, m2);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// ewUnaryMat
// ****************************************************************************

template<class DT>
void checkEwUnaryMat(UnaryOpCode opCode, DT * arg, const DT * exp) {
    DT * res = nullptr;
    ewUnaryMat<DT, DT>(opCode, res, arg, false, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ewUnaryMat - In-Place", TAG_INPLACE, (DenseMatrix), (float)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(4, {
            1, 9, 0, 0, 1, 49,
            0, 36, 0, 9, 0, 0,
            0, 0, 0, 0, 16, 0,
            0, 0, 25, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, { 
            1, 3, 0, 0, 1, 7,
            0, 6, 0, 3, 0, 0,
            0, 0, 0, 0, 4, 0,
            0, 0, 5, 0, 0, 0
    });


    checkEwUnaryMat(UnaryOpCode::SQRT, m1, m2);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}

// ****************************************************************************
// transpose
// ****************************************************************************

template<class DT>
void checkTranspose(DT * arg, const DT * exp) {
    DT * res = nullptr;
    transpose<DT, DT>(res, arg, false, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("Transpose - In-Place", TAG_INPLACE, (DenseMatrix), (uint32_t)) {
    using DT = TestType;
    
    auto m1 = genGivenVals<DT>(3, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
    });
    auto m2 = genGivenVals<DT>(4, { 
            1, 5, 9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12
    });

    checkTranspose(m1, m2);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
}