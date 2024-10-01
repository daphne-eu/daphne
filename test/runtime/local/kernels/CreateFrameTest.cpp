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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CreateFrame.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEST_CASE("CreateFrame", TAG_KERNELS) {
    const size_t numRows = 4;
    const size_t numCols = 3;

    using VT0 = int64_t;
    using VT1 = double;
    using VT2 = float;

    auto c0 = genGivenVals<DenseMatrix<VT0>>(numRows, {1, 2, 3, 4});
    auto c1 = genGivenVals<DenseMatrix<VT1>>(numRows, {-1.2, 3.4, -5.6, 7.8});
    auto c2 = genGivenVals<DenseMatrix<VT2>>(numRows, {11.1, 22.2, 33.3, 44.4});

    Frame *f = nullptr;
    Structure *colMats[] = {c0, c1, c2};
    SECTION("without column labels") { createFrame(f, colMats, numCols, nullptr, 0, nullptr); }
    SECTION("with column labels") {
        const char *labels[] = {"ab", "cde", "fghi"};
        createFrame(f, colMats, numCols, labels, numCols, nullptr);

        // Check column data, access by label.
        CHECK(*(f->template getColumn<VT0>("ab")) == *c0);
        CHECK(*(f->template getColumn<VT1>("cde")) == *c1);
        CHECK(*(f->template getColumn<VT2>("fghi")) == *c2);
    }

    // Check #rows and #cols.
    REQUIRE(f->getNumRows() == numRows);
    REQUIRE(f->getNumCols() == numCols);
    // Check schema.
    auto schema = f->getSchema();
    CHECK(schema[0] == ValueTypeUtils::codeFor<VT0>);
    CHECK(schema[1] == ValueTypeUtils::codeFor<VT1>);
    CHECK(schema[2] == ValueTypeUtils::codeFor<VT2>);
    // Check column data, access by position.
    CHECK(*(f->template getColumn<VT0>(0)) == *c0);
    CHECK(*(f->template getColumn<VT1>(1)) == *c1);
    CHECK(*(f->template getColumn<VT2>(2)) == *c2);

    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(f);
}