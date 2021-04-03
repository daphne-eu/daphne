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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEST_CASE("Frame allocates enough space", TAG_DATASTRUCTURES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.
    
    const size_t numRows = 10000;
    const ValueTypeCode schema[] = {ValueTypeCode::SI8, ValueTypeCode::UI32, ValueTypeCode::F64};
    const size_t numCols = sizeof(schema) / sizeof(ValueTypeCode);
    
    Frame * f = DataObjectFactory::create<Frame>(numRows, numCols, schema, false);
    
    int8_t   * col0 = f->getColumn<int8_t>  (0)->getValues();
    uint32_t * col1 = f->getColumn<uint32_t>(1)->getValues();
    double   * col2 = f->getColumn<double>  (2)->getValues();
    
    // Fill the column arrays with ones of the respective value type.
    for(size_t i = 0; i < numRows; i++) {
        col0[i] = int8_t(1);
        col1[i] = uint32_t(1);
        col2[i] = double(1);
    }
    
    DataObjectFactory::destroy(f);
}

TEST_CASE("Frame sub-frame works properly", TAG_DATASTRUCTURES) {
    const size_t numRowsOrig = 10;
    const ValueTypeCode schemaOrig[] = {ValueTypeCode::SI8, ValueTypeCode::UI32, ValueTypeCode::F64};
    const size_t numColsOrig = sizeof(schemaOrig) / sizeof(ValueTypeCode);
    
    Frame * fOrig = DataObjectFactory::create<Frame>(numRowsOrig, numColsOrig, schemaOrig, true);
    const size_t colIdxsSub[] = {2, 0};
    const size_t numColsSub = sizeof(colIdxsSub) / sizeof(size_t);
    Frame * fSub = DataObjectFactory::create<Frame>(fOrig, 3, 5, numColsSub, colIdxsSub);
    
    // Sub-frame dimensions are as expected.
    CHECK(fSub->getNumRows() == 2);
    CHECK(fSub->getNumCols() == numColsSub);
    
    // Sub-frame schema is as expected.
    CHECK(fSub->getColumnType(0) == ValueTypeCode::F64);
    CHECK(fSub->getColumnType(1) == ValueTypeCode::SI8);

    // Sub-frame shares data arrays with original.
    int8_t * colOrig0 = fOrig->getColumn<int8_t>(0)->getValues();
    double * colOrig2 = fOrig->getColumn<double>(2)->getValues();
    double * colSub0 = fSub->getColumn<double>(0)->getValues();
    int8_t * colSub1 = fSub->getColumn<int8_t>(1)->getValues();
    CHECK((colSub0 >= colOrig2 && colSub0 < colOrig2 + numRowsOrig));
    CHECK((colSub1 >= colOrig0 && colSub1 < colOrig0 + numRowsOrig));
    colSub0[0] = double(123);
    colSub1[0] = int8_t(456);
    CHECK(colOrig2[3] == double(123));
    CHECK(colOrig0[3] == int8_t(456));
    
    // Freeing both frames does not result in double-free errors.
    SECTION("Freeing the original frame first is fine") {
        DataObjectFactory::destroy(fOrig);
        DataObjectFactory::destroy(fSub);
    }
    SECTION("Freeing the sub-frame first is fine") {
        DataObjectFactory::destroy(fSub);
        DataObjectFactory::destroy(fOrig);
    }
}