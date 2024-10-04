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
#include <runtime/local/kernels/InsertCol.h>

#include <catch.hpp>
#include <tags.h>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTArg, typename VTSel>
void checkInsertCol(const DTArg *arg, const DTArg *ins, const VTSel lowerIncl, const VTSel upperExcl,
                    const DTArg *exp) {
    DTArg *res = nullptr;
    insertCol<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res, exp);
}

template <typename DTArg, typename VTSel>
void checkInsertColThrow(const DTArg *arg, const DTArg *ins, const VTSel lowerIncl, const VTSel upperExcl) {
    DTArg *res = nullptr;
    REQUIRE_THROWS_AS((insertCol<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr)),
                      std::out_of_range);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertCol", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
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

    auto ins = genGivenVals<DT>(3, {
                                       1,
                                       -1,
                                       1,
                                       1,
                                       1,
                                       -1,
                                   });

    SECTION("multiple insertions, lower bound") {
        VT lowerIncl = 0;
        VT upperExcl = 2;
        DT *exp = genGivenVals<DT>(3, {
                                          1,
                                          -1,
                                          3,
                                          4,
                                          1,
                                          1,
                                          7,
                                          -8,
                                          1,
                                          -1,
                                          -11,
                                          12,
                                      });

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        VT lowerIncl = 1;
        VT upperExcl = 3;
        DT *exp = genGivenVals<DT>(3, {
                                          1,
                                          1,
                                          -1,
                                          4,
                                          -5,
                                          1,
                                          1,
                                          -8,
                                          9,
                                          1,
                                          -1,
                                          12,
                                      });

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        VT lowerIncl = 2;
        VT upperExcl = 4;
        DT *exp = genGivenVals<DT>(3, {
                                          1,
                                          -2,
                                          1,
                                          -1,
                                          -5,
                                          6,
                                          1,
                                          1,
                                          9,
                                          10,
                                          1,
                                          -1,
                                      });

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("out of bounds - negative") {
        VT lowerIncl = -1;
        VT upperExcl = 1;

        checkInsertColThrow(arg, ins, lowerIncl, upperExcl);
    }

    SECTION("out of bounds - too high") {
        VT lowerIncl = 3;
        VT upperExcl = 5;

        checkInsertColThrow(arg, ins, lowerIncl, upperExcl);
    }

    DataObjectFactory::destroy(arg, ins);
}
TEMPLATE_PRODUCT_TEST_CASE("InsertCol", TAG_KERNELS, DenseMatrix, (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {VT("a"), VT(""), VT("1"), VT("abc"), VT("abc"), VT("abcd"), VT(" "), VT("a"),
                                    VT("ABC"), VT("34ab"), VT("ac"), VT("b")});

    auto ins = genGivenVals<DT>(3, {VT("a"), VT("d"), VT("b"), VT("e"), VT("c"), VT("f")});

    SECTION("multiple insertions, lower bound") {
        size_t lowerIncl = 0;
        size_t upperExcl = 2;
        DT *exp = genGivenVals<DT>(3, {VT("a"), VT("d"), VT("1"), VT("abc"), VT("b"), VT("e"), VT(" "), VT("a"),
                                       VT("c"), VT("f"), VT("ac"), VT("b")});

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        size_t lowerIncl = 1;
        size_t upperExcl = 3;
        DT *exp = genGivenVals<DT>(3, {VT("a"), VT("a"), VT("d"), VT("abc"), VT("abc"), VT("b"), VT("e"), VT("a"),
                                       VT("ABC"), VT("c"), VT("f"), VT("b")});

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        size_t lowerIncl = 2;
        size_t upperExcl = 4;
        DT *exp = genGivenVals<DT>(3, {VT("a"), VT(""), VT("a"), VT("d"), VT("abc"), VT("abcd"), VT("b"), VT("e"),
                                       VT("ABC"), VT("34ab"), VT("c"), VT("f")});

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("out of bounds - negative") {
        size_t lowerIncl = -1;
        size_t upperExcl = 1;

        checkInsertColThrow(arg, ins, lowerIncl, upperExcl);
    }

    SECTION("out of bounds - too high") {
        size_t lowerIncl = 3;
        size_t upperExcl = 5;

        checkInsertColThrow(arg, ins, lowerIncl, upperExcl);
    }

    DataObjectFactory::destroy(arg, ins);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertCol - FP specific", TAG_KERNELS, (DATA_TYPES), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto arg = genGivenVals<DT>(3, {
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

    auto ins = genGivenVals<DT>(3, {
                                       1,
                                       -1,
                                       1,
                                       1,
                                       1,
                                       -1,
                                   });

    SECTION("multiple insertions, FP bounds") {
        VT lowerIncl = 2.4;
        VT upperExcl = 4.9;
        DT *exp = genGivenVals<DT>(3, {
                                          1,
                                          -2,
                                          1,
                                          -1,
                                          -5,
                                          6,
                                          1,
                                          1,
                                          9,
                                          10,
                                          1,
                                          -1,
                                      });

        checkInsertCol(arg, ins, lowerIncl, upperExcl, exp);
    }

    DataObjectFactory::destroy(arg, ins);
}