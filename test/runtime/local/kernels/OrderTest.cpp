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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/Order.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

TEMPLATE_TEST_CASE("Order", TAG_KERNELS, (Frame)) {
    using VT0 = double;
    using VT1 = float;
    using VT2 = int64_t;
    using VT3 = size_t;

    size_t numRows = 20;

    auto c0 = genGivenVals<DenseMatrix<VT0>>(numRows, { 1.1, -3.1, 4.4, -8.8, 5.6, 2.3, 0.3, 4.4, 6.6, 6.6,
                                                        -8.8, 6.6, 6.6, 4.4, -0.3, 4.4, 6.6, 2.3, 6.6, 5.6 });
    auto c1 = genGivenVals<DenseMatrix<VT1>>(numRows, { 1.1, -2, 4.4, 2.1, 1.1, 2.3, 0.5, 4.4, -10, 0, 2.1,
                                                        10, 10, 4.4, -0.3, -15.5, 10, -2.3, 10, -1.1 });
    auto c2 = genGivenVals<DenseMatrix<VT2>>(numRows, { 0, 0, 1, 1, 0, 0, 0, 3, 0, 0, 2, 1, 2, 3, 0, 0, 3, 0, 3, 0 });
    auto c3 = genGivenVals<DenseMatrix<VT3>>(numRows, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 });
    
    std::vector<Structure *> colsArg = {c0, c1, c2, c3};
    auto arg = DataObjectFactory::create<Frame>(colsArg, nullptr);
    DataObjectFactory::destroy(c0, c1, c1, c2);
    Frame* exp{};
    Frame* res{};
    DenseMatrix<size_t>* resIdxs  = nullptr;
    size_t numKeyCols;
    size_t colIdxs[4];
    bool ascending[4];

    DenseMatrix<VT0> * c0Exp{};
    DenseMatrix<VT1> * c1Exp{};
    DenseMatrix<VT2> * c2Exp{};
    DenseMatrix<VT3> * c3Exp{};
    SECTION("single key column, ascending") {
        c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { -8.8, -8.8, -3.1, -0.3, 0.3, 1.1, 2.3, 2.3, 4.4, 4.4,
                                                               4.4, 4.4, 5.6, 5.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6 });
        c1Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 2.1, 2.1, -2, -0.3, 0.5, 1.1, 2.3, -2.3, 4.4, 4.4,
                                                               4.4, -15.5, 1.1, -1.1, -10, 0, 10, 10, 10, 10 });
        c2Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 1, 2, 0, 0, 0, 0, 0, 0, 1, 3, 3, 0, 0, 0, 0, 0, 1, 2, 3, 3 });
        c3Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 3, 10, 1, 14, 6, 0, 5, 17, 2, 7, 13, 15, 4, 19, 8, 9 , 11, 12, 16, 18 });
        numKeyCols = 1;
        colIdxs[0] = 0;
        ascending[0] = true;
    }
    SECTION("single key column, descending") {
        c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 6.6, 6.6, 6.6, 6.6, 4.4, 4.4, 4.4, 2.3, -8.8, -8.8, 
                                                               1.1, 5.6, 0.3, 6.6, -0.3, 5.6, -3.1, 2.3, 6.6, 4.4 });
        c1Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 10, 10, 10, 10, 4.4, 4.4, 4.4, 2.3, 2.1, 2.1, 1.1,
                                                               1.1, 0.5, 0, -0.3, -1.1, -2, -2.3, -10, -15.5 });
        c2Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 1, 2, 3, 3, 1, 3, 3, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        c3Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 11, 12, 16, 18, 2, 7, 13, 5, 3, 10, 0, 4, 6, 9, 14, 19, 1, 17, 8, 15 });
        numKeyCols = 1;
        colIdxs[0] = 1;
        ascending[0] = false;
        
    }
    SECTION("two key columns, ascending/descending") {
        c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { -8.8, -8.8, -3.1, -0.3, 0.3, 1.1, 2.3, 2.3, 4.4, 4.4,
                                                                4.4, 4.4, 5.6, 5.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6 });
        c1Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { 2.1, 2.1, -2, -0.3, 0.5, 1.1, 2.3, -2.3, 4.4, 4.4,
                                                               4.4, -15.5, 1.1, -1.1, 10, 10, 10, 10, -10, 0 });
        c2Exp = genGivenVals<DenseMatrix<VT2>>(numRows, { 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 1, 0, 0, 0, 3, 3, 2, 1, 0, 0 });
        c3Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 10, 3, 1, 14, 6, 0, 5, 17, 7, 13, 2, 15, 4, 19, 16, 18, 12, 11, 8, 9 });
        numKeyCols = 2;
        colIdxs[0] = 0;
        ascending[0] = true;
        colIdxs[1] = 2;
        ascending[1] = false;
    }
    SECTION("four key columns, ascending/descending") {
        c0Exp = genGivenVals<DenseMatrix<VT0>>(numRows, { 4.4, 6.6, 2.3, -3.1, 5.6, -0.3, 6.6, 0.3, 5.6, 1.1,
                                                               -8.8, -8.8, 2.3, 4.4, 4.4, 4.4, 6.6, 6.6, 6.6, 6.6 });
        c1Exp = genGivenVals<DenseMatrix<VT1>>(numRows, { -15.5, -10, -2.3, -2, -1.1, -0.3, 0, 0.5, 1.1, 1.1, 
                                                                2.1, 2.1, 2.3, 4.4, 4.4, 4.4, 10, 10, 10, 10 });
        c2Exp = genGivenVals<DenseMatrix<VT2>>(numRows, {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 3, 1, 3, 3, 2, 1 });
        c3Exp = genGivenVals<DenseMatrix<VT3>>(numRows, { 15, 8, 17, 1, 19, 14, 9, 6, 4, 0, 10, 3, 5, 7, 13, 2, 16, 18, 12, 11 });
        numKeyCols = 4;
        colIdxs[0] = 1;
        ascending[0] = true;
        colIdxs[1] = 0;
        ascending[1] = false;
        colIdxs[2] = 2;
        ascending[2] = false;
        colIdxs[3] = 3;
        ascending[3] = true;
    }

    std::vector<Structure *> colsExp = {c0Exp, c1Exp, c2Exp, c3Exp};
    exp = DataObjectFactory::create<Frame>(colsExp, nullptr);
    DataObjectFactory::destroy(c0Exp, c1Exp, c2Exp);
    
    order(res, arg, colIdxs, numKeyCols, ascending, numKeyCols, false, nullptr);
    order(resIdxs, arg, colIdxs, numKeyCols, ascending, numKeyCols, true, nullptr);
    
    CHECK(*res == *exp);
    CHECK(*resIdxs == *c3Exp);

    DataObjectFactory::destroy(c3Exp, arg, exp, res);
}

TEMPLATE_PRODUCT_TEST_CASE("Order", TAG_KERNELS, (DenseMatrix, Matrix), (double, float)){ // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using DTIdx = typename DT::template WithValueType<size_t>;

    size_t numKeyCols;
    size_t colIdxs[4];
    bool ascending[4];

    DT * argMatrix = nullptr;
    DT * resMatrix = nullptr;
    DT * expMatrix = nullptr;
    DTIdx * resIdxs = nullptr;
    DTIdx * expIdxs = nullptr;

    SECTION("single key column, ascending") {
        argMatrix = genGivenVals<DT>(4, {
            1,  10, 3, 7, 7, 7,
            17, 7,  2, 3, 7, 7,
            7,  7,  1, 2, 3, 7,
            7,  7,  1, 1, 2, 3
        });
        expMatrix = genGivenVals<DT>(4, {
            7,  7,  1, 1, 2, 3,
            7,  7,  1, 2, 3, 7,
            17, 7,  2, 3, 7, 7,
            1,  10, 3, 7, 7, 7
        });
        expIdxs = genGivenVals<DTIdx>(4, { 3, 2, 1, 0});
        numKeyCols = 1;
        colIdxs[0] = 3;
        ascending[0] = true;
    }
    SECTION("four key columns, ascending/descending") {
        argMatrix = genGivenVals<DT>(20, {
            1.1,    1.1,   0,  6,
            -3.1,   -2,    0,  3,
            4.4,    4.4,   1,  9,
            -8.8,   2.1,   1,  1,
            5.6,    1.1,   0,  13,
            2.3,    2.3,   0,  7,
            0.3,    0.5,   0,  5,
            4.4,    4.4,   3,  10,
            6.6,    -10,   0,  15,
            6.6,    0,     0,  16,
            -8.8,   2.1,   2,  2,
            6.6,    10,    1,  17,
            6.6,    10,    2,  18,
            4.4,    4.4,   3,  11,
            -0.3,   -0.3,  0,  4,
            4.4,    -15.5, 0,  12,
            6.6,    10,    3,  19,
            2.3,    -2.3,  0,  8,
            6.6,    10,    3,  20,
            5.6,    -1.1,  0,  14
        });
        expMatrix = genGivenVals<DT>(20, {
            4.4,    -15.5, 0,  12,
            6.6,    -10,   0,  15,
            2.3,    -2.3,  0,  8,
            -3.1,   -2,    0,  3,
            5.6,    -1.1,  0,  14,
            -0.3,   -0.3,  0,  4,
            6.6,    0,     0,  16,
            0.3,    0.5,   0,  5,
            5.6,    1.1,   0,  13,
            1.1,    1.1,   0,  6,
            -8.8,   2.1,   2,  2,
            -8.8,   2.1,   1,  1,
            2.3,    2.3,   0,  7,
            4.4,    4.4,   3,  10,
            4.4,    4.4,   3,  11,
            4.4,    4.4,   1,  9,
            6.6,    10,    3,  19,
            6.6,    10,    3,  20,
            6.6,    10,    2,  18,
            6.6,    10,    1,  17
        });
        expIdxs = genGivenVals<DTIdx>(20, { 15, 8, 17, 1, 19, 14, 9, 6, 4, 0,
                                            10, 3, 5, 7, 13, 2, 16, 18, 12, 11 });
        numKeyCols = 4;
        colIdxs[0] = 1;
        ascending[0] = true;
        colIdxs[1] = 0;
        ascending[1] = false;
        colIdxs[2] = 2;
        ascending[2] = false;
        colIdxs[3] = 3;
        ascending[3] = true;
    }

    order(resMatrix, argMatrix, colIdxs, numKeyCols, ascending, numKeyCols, false, nullptr);
    order(resIdxs, argMatrix, colIdxs, numKeyCols, ascending, numKeyCols, true, nullptr);

    CHECK(*resMatrix == *expMatrix);
    CHECK(*resIdxs == *expIdxs);

    DataObjectFactory::destroy(argMatrix, resMatrix, expMatrix, resIdxs, expIdxs);
}