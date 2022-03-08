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
#include <runtime/local/kernels/SliceRow.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("SliceRow", TAG_KERNELS, DenseMatrix, (double, int64_t, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 1, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6, 7,
        0, 8, 0, 0, 9, 0,
    };
    std::vector<typename DT::VT> valsExp = {
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6, 7,
    };
    auto arg = genGivenVals<DT>(4, vals);
    auto exp = genGivenVals<DT>(2, valsExp);
    DT * res = nullptr;
    sliceRow(res, arg, 1, 3, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_TEST_CASE("SliceRow", TAG_KERNELS, (Frame)) {
    using VT = double;

    auto c0 = genGivenVals<DenseMatrix<VT>>(4, {0.0, 1.1, 2.2, 3.3});
    auto c1 = genGivenVals<DenseMatrix<VT>>(4, {4.4, 5.5, 6.6, 7.7});
    auto c2 = genGivenVals<DenseMatrix<VT>>(4, {8.8, 9.9, 1.0, 2.0});
    auto c3 = genGivenVals<DenseMatrix<VT>>(4, {3.0, 4.0, 5.0, 6.0});
    auto c4 = genGivenVals<DenseMatrix<VT>>(2, {1.1, 2.2});
    auto c5 = genGivenVals<DenseMatrix<VT>>(2, {5.5, 6.6});
    auto c6 = genGivenVals<DenseMatrix<VT>>(2, {9.9, 1.0});
    auto c7 = genGivenVals<DenseMatrix<VT>>(2, {4.0, 5.0});
    std::vector<Structure *> cols1 = {c0, c1, c2, c3};
    std::vector<Structure *> cols2 = {c4, c5, c6, c7};
    auto arg = DataObjectFactory::create<Frame>(cols1, nullptr);
    auto exp = DataObjectFactory::create<Frame>(cols2, nullptr);
    Frame * res = nullptr;
    sliceRow(res, arg, 1, 3, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}