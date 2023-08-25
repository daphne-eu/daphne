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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("CSRMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.

    using ValueType = TestType;

    const size_t numRows = 10000;
    const size_t numCols = 2000;
    const size_t numNonZeros = 500;

    CSRMatrix<ValueType> * m = DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, false);

    ValueType * values = m->getValues();
    size_t * colIdxs = m->getColIdxs();
    size_t * rowOffsets = m->getRowOffsets();

    // Fill all arrays with ones of the respective type. Note that this does
    // not result in a valid CSR representation, but we only want to check if
    // there is enough space.
    for(size_t i = 0; i < numNonZeros; i++) {
        values[i] = ValueType(1);
        colIdxs[i] = size_t(1);
    }
    for(size_t i = 0; i <= numRows; i++)
        rowOffsets[i] = size_t(1);

    


    DataObjectFactory::destroy(m);
}

TEST_CASE("CSRMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;

    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 7;
    const size_t numNonZeros = 3;

    CSRMatrix<ValueType> * mOrig = DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsOrig, numColsOrig, numNonZeros, true);
    CSRMatrix<ValueType> * mSub = DataObjectFactory::create<CSRMatrix<ValueType>>(mOrig, 3, 5);

    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == numColsOrig);

    // Sub-matrix shares data array with original.
    CHECK(mSub->getValues() == mOrig->getValues());
    CHECK(mSub->getColIdxs() == mOrig->getColIdxs());

    ValueType * rowOffsetsOrig = mOrig->getRowOffsets();
    ValueType * rowOffsetsSub = mSub->getRowOffsets();
    CHECK((rowOffsetsSub >= rowOffsetsOrig && rowOffsetsSub <= rowOffsetsOrig + numRowsOrig));
    rowOffsetsSub[0] = 123;
    CHECK(rowOffsetsOrig[3] == 123);

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
