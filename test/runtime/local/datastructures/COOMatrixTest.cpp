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

#include <runtime/local/datastructures/COOMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("COOMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.

    using ValueType = TestType;

    const size_t numRows = 10000;
    const size_t numCols = 2000;
    const size_t numNonZeros = 500;

    COOMatrix<ValueType> * m = DataObjectFactory::create<COOMatrix<ValueType>>(numRows, numCols, numNonZeros, false);

    ValueType * values = m->getValues();
    size_t * colIdxs = m->getColIdxs();
    size_t * rowIdxs = m->getRowIdxs();

    // Fill all arrays with ones of the respective type. Note that this does
    // not result in a valid COO representation, but we only want to check if
    // there is enough space.
    for(size_t i = 0; i < numNonZeros; i++) {
        values[i] = ValueType(1);
        colIdxs[i] = size_t(1);
        rowIdxs[i] = size_t(1);
    }

    DataObjectFactory::destroy(m);
}

TEST_CASE("COOMatrix methods work properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;

    const size_t numRows = 10;
    const size_t numCols = 10;
    const size_t maxnumNonZeros = 6;

    COOMatrix<ValueType> * matrix = DataObjectFactory::create<COOMatrix<ValueType>>(numRows, numCols, maxnumNonZeros, true);

    matrix->set(0, 0, 5);
    matrix->set(2, 2, 3);
    matrix->set(1, 1, 4);
    matrix->set(3, 3, 2);
    matrix->set(4, 4, 1);

    CHECK(matrix->getMaxNumNonZeros() == 6);
    CHECK(matrix->getNumNonZeros() == 5);
    CHECK(matrix->getNumRows() == 10);
    CHECK(matrix->getNumCols() == 10);
    CHECK(matrix->getNumNonZerosRow(1) == 1);
    CHECK(matrix->getNumNonZerosCol(1) == 1);
    CHECK(matrix->getValues()[0] == 5);
    CHECK(matrix->getColIdxs()[3] == 3);
    CHECK(matrix->getRowIdxs()[2] == 2);
    CHECK(matrix->getValues(1)[0] == 4);
    CHECK(matrix->getColIdxs(1)[0] == 1);
    CHECK(matrix->get(1, 1) == 4);

    matrix->prepareAppend();
    matrix->append(0, 0, 5);
    matrix->append(1, 1, 4);
    matrix->append(2, 2, 3);
    matrix->append(3, 3, 2);
    matrix->append(4, 4, 1);

    CHECK(matrix->getNumNonZeros() == 5);
    CHECK(matrix->getNumNonZerosRow(1) == 1);
    CHECK(matrix->getNumNonZerosCol(1) == 1);
    CHECK(matrix->getValues()[0] == 5);
    CHECK(matrix->getColIdxs()[3] == 3);
    CHECK(matrix->getRowIdxs()[2] == 2);
    CHECK(matrix->getValues(1)[0] == 4);
    CHECK(matrix->getColIdxs(1)[0] == 1);
    CHECK(matrix->get(1, 1) == 4);
}

TEST_CASE("COOMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;

    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 10;
    const size_t maxnumNonZeros = 10;

    COOMatrix<ValueType> * mOrig = DataObjectFactory::create<COOMatrix<ValueType>>(numRowsOrig, numColsOrig, maxnumNonZeros, true);
    COOMatrix<ValueType> * mSub = DataObjectFactory::create<COOMatrix<ValueType>>(mOrig, 3, 5);

    mOrig->set(0, 0, 5);
    mOrig->set(2, 2, 3);
    mOrig->set(1, 1, 4);
    mOrig->set(3, 3, 2);
    mOrig->set(4, 4, 1);

    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == numColsOrig);

    // Sub-matrix shares arrays with original.
    CHECK(mSub->getValues()[0] == mOrig->getValues()[3]);
    CHECK(mSub->getColIdxs()[0] == mOrig->getColIdxs()[3]);
    CHECK(mSub->getRowIdxs()[0] == mOrig->getRowIdxs()[3]);

    CHECK(mOrig->get(3, 3) == mSub->get(0, 3));
    CHECK(mOrig->get(4, 4) == mSub->get(1, 4));

    mOrig->set(3, 4, 15);

    CHECK(mSub->getNumNonZeros() == 3);

    mOrig->set(0, 1, 15);

    CHECK(mSub->getNumNonZeros() == 3);

    CHECK(mOrig->get(3, 3) == mSub->get(0, 3));
    CHECK(mOrig->get(4, 4) == mSub->get(1, 4));

    // Freeing both matrices does not result in double-free errors.
    SECTION("Freeing the original matrix first is fine") {
        DataObjectFactory::destroy(mOrig);
        DataObjectFactory::destroy(mSub);
    }
    SECTION("Freeing the sub-matrix first is fine") {
        DataObjectFactory::destroy(mSub);
        DataObjectFactory::destroy(mOrig);
    }
}