/*
 * Copyright 2024 The DAPHNE Consortium
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
#include <runtime/local/kernels/Fill.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "Fill (" opName ")"
#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int64_t, double

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Matrix"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    VT arg;
    size_t numRows, numCols;
    DT *exp = nullptr;

    SECTION("2x2 matrix") {
        arg = VT(1.5);
        numRows = 2;
        numCols = 2;

        exp = genGivenVals<DT>(2, {VT(1.5), VT(1.5), VT(1.5), VT(1.5)});
    }
    SECTION("1x5 matrix") {
        arg = VT(2.5);
        numRows = 1;
        numCols = 5;

        exp = genGivenVals<DT>(1, {VT(2.5), VT(2.5), VT(2.5), VT(2.5), VT(2.5)});
    }

    DT *res = nullptr;
    fill(res, arg, numRows, numCols, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(exp, res);
}

TEMPLATE_PRODUCT_TEST_CASE("FillString", TAG_KERNELS, (DenseMatrix), (ALL_STRING_VALUE_TYPES)){
    using DT = TestType;
    using VT = typename DT::VT;

    size_t numRows = 3;
    size_t numCols = 4;

    SECTION("empty_string"){
        DenseMatrix<VT> * res = nullptr;
        VT arg = VT("");

        auto * exp = genGivenVals<DenseMatrix<VT>>(3, {
            VT(""), VT(""), VT(""), VT(""),
            VT(""), VT(""), VT(""), VT(""),
            VT(""), VT(""), VT(""), VT("")
        });

        fill(res, arg, numRows, numCols, nullptr);
        CHECK(*exp == *res);

        DataObjectFactory::destroy(res, exp);
    }

    SECTION("not_empty_string"){
        DenseMatrix<VT> * res = nullptr;
        VT arg = VT("abc");

        auto * exp = genGivenVals<DenseMatrix<VT>>(3, {
            VT("abc"), VT("abc"), VT("abc"), VT("abc"),
            VT("abc"), VT("abc"), VT("abc"), VT("abc"),
            VT("abc"), VT("abc"), VT("abc"), VT("abc")
        });

        fill(res, arg, numRows, numCols, nullptr);
        CHECK(*exp == *res);

        DataObjectFactory::destroy(res, exp);
    }
}
