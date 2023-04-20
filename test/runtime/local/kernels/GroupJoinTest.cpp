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
#include <runtime/local/kernels/GroupJoin.h>
#include <runtime/local/kernels/Seq.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

/**
 * @brief Runs the groupJoin-kernel with small input data and performs various
 * checks.
 */
TEST_CASE("GroupJoin", TAG_KERNELS) {
    // lhs is a kind of dimension table.
    auto lhsC0 = genGivenVals<DenseMatrix<int64_t>>(3, { 1,  2,  3});
    auto lhsC1 = genGivenVals<DenseMatrix<int64_t>>(3, {11, 22, 33});
    std::vector<Structure *> lhsCols = {lhsC0, lhsC1};
    std::string lhsLabels[] = {"d.id", "d.foo"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    // rhs is a kind of fact table.
    auto rhsC0 = genGivenVals<DenseMatrix<int64_t>>(10, { 1,  1,  1,  3,  1,  3,  3,  1,  3,  3});
    auto rhsC1 = genGivenVals<DenseMatrix<int64_t>>(10, {42, 42, 42, 42, 42, 42, 42, 42, 42, 42});
    auto rhsC2 = genGivenVals<DenseMatrix<double >>(10, {10, 20, 30, 10, 20, 30, 10, 20, 30, 10});
    std::vector<Structure *> rhsCols = {rhsC0, rhsC1, rhsC2};
    std::string rhsLabels[] = {"f.id", "f.bar", "f.agg"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * res = nullptr;
    DenseMatrix<size_t> * lhsTid = nullptr;
    groupJoin<size_t>(res, lhsTid, lhs, rhs, "d.id", "f.id", "f.agg", nullptr);

    // Check the meta data.
    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 2);
    CHECK(res->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(1) == ValueTypeCode::F64);
    CHECK(res->getLabels()[0] == "d.id");
    CHECK(res->getLabels()[1] == "SUM(f.agg)");
    CHECK(lhsTid->getNumRows() == 2);
    CHECK(lhsTid->getNumCols() == 1);
    
    // Check the data.
#if 0
    // TODO Since any order of rows would be correct, we should sort before the
    // comparison, but we do not have a kernel for sorting yet.
    auto resC0Exp = genGivenVals<DenseMatrix<int64_t>>(2, {1, 3});
    auto resC1Exp = genGivenVals<DenseMatrix<double>>(2, {100, 90});
    CHECK(*(res->getColumn<int64_t>(0)) == *resC0Exp);
    CHECK(*(res->getColumn<double >(1)) == *resC1Exp);
    auto lhsTidExp = genGivenVals<DenseMatrix<size_t>>(2, {0, 2});
    CHECK(*lhsTid == *lhsTidExp);
#else
    auto resC0Fnd = res->getColumn<int64_t>(0);
    auto resC1Fnd = res->getColumn<double>(1);
    const bool dataGood = (
        // the one order
        resC0Fnd->get(0, 0) ==   1 && resC0Fnd->get(1, 0) ==  3 &&
        resC1Fnd->get(0, 0) == 100 && resC1Fnd->get(1, 0) == 90 &&
        lhsTid  ->get(0, 0) ==   0 && lhsTid  ->get(1, 0) ==  2
    ) || (
        // the other order
        resC0Fnd->get(1, 0) ==   1 && resC0Fnd->get(0, 0) ==  3 &&
        resC1Fnd->get(1, 0) == 100 && resC1Fnd->get(0, 0) == 90 &&
        lhsTid  ->get(1, 0) ==   0 && lhsTid  ->get(0, 0) ==  2
    );
    CHECK(dataGood);
#endif
}