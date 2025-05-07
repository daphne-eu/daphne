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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ConvertPosListToBitmap.h>

#include <tags.h>

#include <catch.hpp>

TEMPLATE_TEST_CASE("convertPosListToBitmap", TAG_KERNELS, double, int64_t, size_t) {
    using VT = TestType;

    DenseMatrix<VT> *arg = nullptr;
    const size_t numRowsRes = 10;
    DenseMatrix<VT> *exp = nullptr;

    SECTION("empty poslist") {
        arg = DataObjectFactory::create<DenseMatrix<VT>>(0, 1, false);
        exp = genGivenVals<DenseMatrix<VT>>(10, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    SECTION("non-empty poslist, case 1") {
        arg = genGivenVals<DenseMatrix<VT>>(5, {0, 3, 7, 8, 9});
        exp = genGivenVals<DenseMatrix<VT>>(10, {1, 0, 0, 1, 0, 0, 0, 1, 1, 1});
    }
    SECTION("non-empty poslist, case 2") {
        arg = genGivenVals<DenseMatrix<VT>>(5, {2, 3, 4, 5, 6});
        exp = genGivenVals<DenseMatrix<VT>>(10, {0, 0, 1, 1, 1, 1, 1, 0, 0, 0});
    }

    DenseMatrix<VT> *res = nullptr;
    convertPosListToBitmap(res, arg, numRowsRes, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(arg, exp, res);
}