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
#include <runtime/local/kernels/FilterCol.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <type_traits>
#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES double, int64_t, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("FilterCol", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using VTSel = int64_t;
    using DTSel = DenseMatrix<VTSel>;
    using DTEmpty = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;

    auto arg = genGivenVals<DT>(3, {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    });

    DTSel * sel = nullptr;
    DT * exp = nullptr;
    SECTION("bit vector empty") {
        sel = genGivenVals<DTSel>(5, {0, 0, 0, 0, 0});
        exp = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(3, 0, false));
    }
    SECTION("bit vector contiguous 0") {
        sel = genGivenVals<DTSel>(5, {0, 0, 1, 1, 1});
        exp = genGivenVals<DT>(3, {
            3, 4, 5,
            8, 9, 10,
            13, 14, 15
        });
    }
    SECTION("bit vector contiguous 1") {
        sel = genGivenVals<DTSel>(5, {1, 1, 1, 0, 0});
        exp = genGivenVals<DT>(3, {
            1, 2, 3,
            6, 7, 8,
            11, 12, 13
        });
    }
    SECTION("bit vector mixed") {
        sel = genGivenVals<DTSel>(5, {0, 1, 1, 0, 1});
        exp = genGivenVals<DT>(3, {
            2, 3, 5,
            7, 8, 10,
            12, 13, 15
        });
    }
    SECTION("bit vector full") {
        sel = genGivenVals<DTSel>(5, {1, 1, 1, 1, 1});
        exp = genGivenVals<DT>(3, {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15
        });
    }

    DT * res = nullptr;
    filterCol<DT, DT, VTSel>(res, arg, sel, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(arg, sel, exp, res);
}