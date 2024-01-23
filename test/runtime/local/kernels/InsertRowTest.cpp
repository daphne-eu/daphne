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

#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/InsertRow.h>

#include <tags.h>
#include <catch.hpp>

#include <cstdint>

#define VALUE_TYPES int32_t, double

template<typename DTArg, typename VTSel>
void checkInsertRow(const DTArg * arg, const DTArg * ins, const VTSel lowerIncl, const VTSel upperExcl, const DTArg * exp) {
    DTArg * res = nullptr;
    insertRow<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res, exp);
}

template<typename DTArg, typename VTSel>
void checkInsertRowThrow(const DTArg * arg, const DTArg * ins, const VTSel lowerIncl, const VTSel upperExcl) {
    DTArg * res = nullptr;
    REQUIRE_THROWS_AS((insertRow<DTArg, DTArg, VTSel>(res, arg, ins, lowerIncl, upperExcl, nullptr)), std::out_of_range);
}

TEMPLATE_PRODUCT_TEST_CASE("InsertRow", TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;


    auto arg = genGivenVals<DT>(4, {
        1, -2, 3,
        4, -5, 6,
        7, -8, 9,
        10, -11, 12,
    });

    auto ins = genGivenVals<DT>(2, {
        2, -2, 2,
        7, 9, 11,
    });

    SECTION("multiple insertions, lower bound") {
        VT lowerIncl = 0;
        VT upperExcl = 2;
        DT * exp = genGivenVals<DT>(4, {
            2, -2, 2,
            7, 9, 11,
            7, -8, 9,
            10, -11, 12,
        });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertion, middle") {
        VT lowerIncl = 1;
        VT upperExcl = 3;
        DT * exp = genGivenVals<DT>(4, {
            1, -2, 3,
            2, -2, 2,
            7, 9, 11,
            10, -11, 12,
        });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    SECTION("multiple insertions, upper bound") {
        VT lowerIncl = 2;
        VT upperExcl = 4;
        DT * exp = genGivenVals<DT>(4, {
            1, -2, 3,
            4, -5, 6,
            2, -2, 2,
            7, 9, 11,
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

TEMPLATE_PRODUCT_TEST_CASE("InsertRow - FP specific", TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;


    auto arg = genGivenVals<DT>(4, {
        1, -2, 3,
        4, -5, 6,
        7, -8, 9,
        10, -11, 12.4,
    });

    auto ins = genGivenVals<DT>(2, {
        2, -2, 2,
        7, 9, 11,
    });
    
    SECTION("multiple insertions, FP bounds") {
        VT lowerIncl = 2.4;
        VT upperExcl = 4.9;
        DT * exp = genGivenVals<DT>(4, {
            1, -2, 3,
            4, -5, 6,
            2, -2, 2,
            7, 9, 11,
        });

        checkInsertRow(arg, ins, lowerIncl, upperExcl, exp);
    }

    DataObjectFactory::destroy(arg, ins);
}