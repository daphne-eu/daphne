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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/SemiJoin.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEST_CASE("SemiJoin", TAG_KERNELS) {
    // lhs
    auto lhsC0 = genGivenVals<DenseMatrix<int64_t>>(4, {1, 2, 3, 4});
    auto lhsC1 = genGivenVals<DenseMatrix<double>>(4, {11.0, 22.0, 33.0, 44.00});
    std::vector<Structure *> lhsCols = {lhsC0, lhsC1};
    std::string lhsLabels[] = {"a", "b"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    // rhs
    auto rhsC0 = genGivenVals<DenseMatrix<int64_t>>(3, {1, 4, 5});
    auto rhsC1 = genGivenVals<DenseMatrix<int64_t>>(3, {-1, -4, -5});
    auto rhsC2 = genGivenVals<DenseMatrix<double>>(3, {0.1, 0.2, 0.3});
    std::vector<Structure *> rhsCols = {rhsC0, rhsC1, rhsC2};
    std::string rhsLabels[] = {"c", "d", "e"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);

    // expRes
    auto expResC0 = genGivenVals<DenseMatrix<int64_t>>(2, {1, 4});
    std::vector<Structure *> expResCols = {expResC0};
    std::string expResLabels[] = {"a"};
    auto expRes = DataObjectFactory::create<Frame>(expResCols, expResLabels);

    // expTid
    auto expTid = genGivenVals<DenseMatrix<int64_t>>(2, {0, 3});

    // res
    Frame *res = nullptr;
    DenseMatrix<int64_t> *lhsTid = nullptr;
    semiJoin(res, lhsTid, lhs, rhs, "a", "c", -1, nullptr);

    CHECK(*res == *expRes);
    CHECK(*lhsTid == *expTid);

    DataObjectFactory::destroy(lhs, rhs, expRes, expTid, res, lhsTid, lhsC0, lhsC1, rhsC0, rhsC1, rhsC2, expResC0);
}