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

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Tri.h>

#include <tags.h>

#include <catch.hpp>

#define TEST_NAME(opName) "Tri (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkTri(const DT * arg, const DT * exp, bool upper, bool diag, bool values) {
    DT * res = nullptr;
    tri<DT>(res, arg, upper, diag, values, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("example"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(4, {
        1, 0, 2, 3,
        4, 5, 0, 6,
        0, 7, 0, 8,
        0, 0, 9, 0,
    });

    auto m1 = genGivenVals<DT>(4, {
        1, 0, 0, 0,
        4, 5, 0, 0,
        0, 7, 0, 0,
        0, 0, 9, 0,
    });

    auto m2 = genGivenVals<DT>(4, {
        0, 0, 1, 1,
        0, 0, 0, 1,
        0, 0, 0, 1,
        0, 0, 0, 0,
    });

    checkTri(m, m1, false, true, true);
    checkTri(m, m2, true, false, false);

    DataObjectFactory::destroy(m, m1, m2);
}
