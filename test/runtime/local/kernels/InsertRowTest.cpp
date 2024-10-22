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
#include <runtime/local/kernels/InsertRow.h>

#include <catch.hpp>
#include <tags.h>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTArg, typename VTSel>
void checkInsertRow(const DTArg *arg, const DTArg *ins, const VTSel lowerIncl, const VTSel upperExcl,
                    const DTArg *exp) {
    DTArg *res = nullptr;
    insertRow<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res, exp);
}

template <typename DTArg, typename VTSel>
void checkInsertRowThrow(const DTArg *arg, const DTArg *ins, const VTSel lowerIncl, const VTSel upperExcl) {
    DTArg *res = nullptr;
    REQUIRE_THROWS_AS((insertRow<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr)),
                      std::out_of_range);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(4, {
                                       1,
                                       -2,
                                       3,
                                       4,
                                       -5,
                                       6,
                                       7,
                                       -8,
                                       9,
                                       10,
                                       -11,
                                       12,
                                   });

    auto ins = genGivenVals<DT>(2, {
                                       2,
                                       -2,
                                       2,
                                       7,
                                       9,
                                       11,
                                   });

    SECTION("multiple insertions, lower bound") {
        VT lowerIncl = 0;
        VT upperExcl = 2;
        DT *exp = genGivenVals<DT>(4, {
                                          2,
                                          -2,
                                          2,
                                          7,
                                          9,
                                          11,
                                          7,
                                          -8,
                                          9,
                                          10,
                                          -11,
                                          12,
                                      });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        VT lowerIncl = 1;
        VT upperExcl = 3;
        DT *exp = genGivenVals<DT>(4, {
                                          1,
                                          -2,
                                          3,
                                          2,
                                          -2,
                                          2,
                                          7,
                                          9,
                                          11,
                                          10,
                                          -11,
                                          12,
                                      });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        VT lowerIncl = 2;
        VT upperExcl = 4;
        DT *exp = genGivenVals<DT>(4, {
                                          1,
                                          -2,
                                          3,
                                          4,
                                          -5,
                                          6,
                                          2,
                                          -2,
                                          2,
                                          7,
                                          9,
                                          11,
                                      });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("out of bounds - negative") {
        VT lowerIncl = -1;
        VT upperExcl = 1;

        checkInsertRowThrow(arg, ins, lowerIncl, upperExcl);
    }

    SECTION("out of bounds - too high") {
        VT lowerIncl = 3;
        VT upperExcl = 5;

        checkInsertRowThrow(arg, ins, lowerIncl, upperExcl);
    }

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - string specific", TAG_KERNELS, (DATA_TYPES), (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("abc"), VT("abc"), VT("abcd"), VT(" "), VT("a"),
                                    VT("ABC"), VT("34ab"), VT("ac"), VT("b")});

    auto ins = genGivenVals<DT>(2, {VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f")});

    SECTION("multiple insertions, lower bound") {
        size_t lowerIncl = 0;
        size_t upperExcl = 2;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f"), VT(" "), VT("a"),
                                       VT("ABC"), VT("34ab"), VT("ac"), VT("b")});

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        size_t lowerIncl = 1;
        size_t upperExcl = 3;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("a"), VT("b"), VT("c"), VT("d"), VT("e"), VT("f"),
                                       VT("34ab"), VT("ac"), VT("b")});

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        size_t lowerIncl = 2;
        size_t upperExcl = 4;
        DT *exp = genGivenVals<DT>(4, {VT("a"), VT(""), VT("1"), VT("abc"), VT("abc"), VT("abcd"), VT("a"), VT("b"),
                                       VT("c"), VT("d"), VT("e"), VT("f")});

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("out of bounds - negative") {
        size_t lowerIncl = -1;
        size_t upperExcl = 1;

        checkInsertRowThrow(arg, ins, lowerIncl, upperExcl);
    }

    SECTION("out of bounds - too high") {
        size_t lowerIncl = 3;
        size_t upperExcl = 5;

        checkInsertRowThrow(arg, ins, lowerIncl, upperExcl);
    }

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - FP specific", TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(4, {
                                       1,
                                       -2,
                                       3,
                                       4,
                                       -5,
                                       6,
                                       7,
                                       -8,
                                       9,
                                       10,
                                       -11,
                                       12.4,
                                   });

    auto ins = genGivenVals<DT>(2, {
                                       2,
                                       -2,
                                       2,
                                       7,
                                       9,
                                       11,
                                   });

    SECTION("multiple insertions, FP bounds") {
        VT lowerIncl = 2.4;
        VT upperExcl = 4.9;
        DT *exp = genGivenVals<DT>(4, {
                                          1,
                                          -2,
                                          3,
                                          4,
                                          -5,
                                          6,
                                          2,
                                          -2,
                                          2,
                                          7,
                                          9,
                                          11,
                                      });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    DataObjectFactory::destroy(arg, ins);
}