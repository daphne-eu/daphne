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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("DenseMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.
    
    using ValueType = TestType;
    
    const size_t numRows = 10000;
    const size_t numCols = 2000;
    
    DenseMatrix<ValueType> * m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);
    
    ValueType * values = m->getValues();
    const size_t numCells = numRows * numCols;
    
    // Fill the matrix with ones of the respective value type.
    for(size_t i = 0; i < numCells; i++)
        values[i] = ValueType(1);
    
    DataObjectFactory::destroy(m);
}

TEST_CASE("DenseMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;
    
    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 7;
    const size_t numCellsOrig = numRowsOrig * numColsOrig;
    
    DenseMatrix<ValueType> * mOrig = DataObjectFactory::create<DenseMatrix<ValueType>>(numRowsOrig, numColsOrig, true);
    DenseMatrix<ValueType> * mSub = DataObjectFactory::create<DenseMatrix<ValueType>>(mOrig, 3, 5, 1, 4);
    
    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == 3);
    CHECK(mSub->getRowSkip() == numColsOrig);

    // Sub-matrix shares data array with original.
    ValueType * valuesOrig = mOrig->getValues();
    ValueType * valuesSub = mSub->getValues();
    CHECK((valuesSub >= valuesOrig && valuesSub < valuesOrig + numCellsOrig));
    valuesSub[0] = 123;
    CHECK(valuesOrig[3 * numColsOrig + 1] == 123);
    
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