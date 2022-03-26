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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/InnerJoin.h>
#include <runtime/local/kernels/Seq.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEST_CASE("innerJoin", TAG_KERNELS) {
    auto lhsC0 = genGivenVals<DenseMatrix<int64_t>>(4, { 1,  2,  3,  4});
    auto lhsC1 = genGivenVals<DenseMatrix<double>>(4, {11.0, 22.0, 33.0, 44.00});
    std::vector<Structure *> lhsCols = {lhsC0, lhsC1};
    std::string lhsLabels[] = {"a", "b"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhsC0 = genGivenVals<DenseMatrix<int64_t>>(3, { 1, 4, 5});
    auto rhsC1 = genGivenVals<DenseMatrix<int64_t>>(3, { -1, -4, -5});
    auto rhsC2 = genGivenVals<DenseMatrix<double >>(3, {0.1, 0.2, 0.3});
    std::vector<Structure *> rhsCols = {rhsC0, rhsC1, rhsC2};
    std::string rhsLabels[] = {"c", "d", "e"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);

    Frame * res = nullptr;
    innerJoin(res, lhs, rhs, "a", "c", nullptr);

    // Check the meta data.
    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 5);

    CHECK(res->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(1) == ValueTypeCode::F64);
    CHECK(res->getColumnType(2) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(3) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(4) == ValueTypeCode::F64);

    CHECK(res->getLabels()[0] == "a");
    CHECK(res->getLabels()[1] == "b");
    CHECK(res->getLabels()[2] == "c");
    CHECK(res->getLabels()[3] == "d");
    CHECK(res->getLabels()[4] == "e");

    auto resC0Exp = genGivenVals<DenseMatrix<int64_t>>(2, {1, 4});
    auto resC1Exp = genGivenVals<DenseMatrix<double >>(2, {11.0, 44.0});
    auto resC2Exp = genGivenVals<DenseMatrix<int64_t>>(2, {1, 4});
    auto resC3Exp = genGivenVals<DenseMatrix<int64_t>>(2, {-1, -4});
    auto resC4Exp = genGivenVals<DenseMatrix<double >>(2, {0.1, 0.2});

    CHECK(*(res->getColumn<int64_t>(0)) == *resC0Exp);
    CHECK(*(res->getColumn<double >(1)) == *resC1Exp);
    CHECK(*(res->getColumn<int64_t>(2)) == *resC2Exp);
    CHECK(*(res->getColumn<int64_t>(3)) == *resC3Exp);
    CHECK(*(res->getColumn<double >(4)) == *resC4Exp);
    
    DataObjectFactory::destroy(lhsC0, lhsC1, lhs);
    DataObjectFactory::destroy(rhsC0, rhsC1, rhsC2, rhs);
    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(resC0Exp, resC1Exp, resC2Exp, resC3Exp, resC4Exp);
}
