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

 #include <runtime/local/datastructures/CSCMatrix.h>
 #include <runtime/local/datastructures/DataObjectFactory.h>
 #include <runtime/local/datastructures/ValueTypeUtils.h>

 #include <tags.h>

 #include <catch.hpp>

 #include <cstdint>

TEMPLATE_TEST_CASE("CSCMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 8;

   CSCMatrix<ValueType> * m = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   CHECK(m->getNumRows() == 4);
   CHECK(m->getNumCols() == 6);
   CHECK(m->getMaxNumNonZeros() == 8);

   DataObjectFactory::destroy(m);

}


TEMPLATE_TEST_CASE("CSCMatrix appends and updates values correctly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 8;

   CSCMatrix<ValueType> * m = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, false);

   //Appending process: prepareAppend() -> append()...append() -> finishAppend()
   m -> prepareAppend();
   //First column
   m -> append(0,0,10);
   //Second column
   m -> append(0,1,20);
   m -> append(1,1,30);
   //Third column
   m -> append(2,2,50);
   //Fourth column
   m -> append(1,3,40);
   m -> append(2,3,60);
   //Fith column
   m -> append(2,4,70);
   //Sixth column
   m -> append(3,5,80);
   m -> finishAppend();

   ValueType * values = m->getValues();
   size_t * rowIdxs = m->getRowIdxs();
   size_t * columnOffsets = m->getColumnOffsets();

   ValueType valuesControl[] = {10, 20, 30, 50, 40, 60, 70, 80};
   size_t rowIdxsControl[] = {0, 0, 1, 2, 1, 2, 2, 3};
   size_t columnOffsetsControl[] = {0, 1, 3, 4, 6, 7, 8};

   for(size_t i = 0; i<maxNumNonZeros; i++){
     CHECK(values[i] == valuesControl[i]);
     CHECK(rowIdxs[i] == rowIdxsControl[i]);
     if(i<numCols+1){
       CHECK(columnOffsets[i] == columnOffsetsControl[i]);
     }
   }

   // After appending values nnz should be 8
   CHECK(m->getNumNonZeros() == 8);
   // Verify get() function by setting row and column coordinates
   CHECK(m->get(2,3) == 60);
   // Set matrix cell in position (1,1) from 30 to 35
   m->set(1,1, 35);
   CHECK(m->getValues()[2]==35);
   // Set matrix cell at position (0,1) from 20 to 0
   // As a result, nnz should be 7 instead of 8 as before
   m->set(0,1,0);
   CHECK((m -> get(0,1) == 0 && m->getNumNonZeros() == 7));



   DataObjectFactory::destroy(m);

}


TEMPLATE_TEST_CASE("CSCMatrix sub-matrix works properly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRowsOrig = 4;
  const size_t numColsOrig = 6;
  const size_t maxNumNonZeros = 8;

  const size_t colLowerIncl = 3;
  const size_t colUpperExcl = 5;

  CSCMatrix<ValueType> * mOrig = DataObjectFactory::create<CSCMatrix<ValueType>>(numRowsOrig, numColsOrig, maxNumNonZeros, true);
  // Create sub-matrix from original matrix
  CSCMatrix<ValueType> * mSub = mOrig -> sliceCol(colLowerIncl, colUpperExcl);

  // Sub-matrix dimensions are as expected.
  CHECK(mSub->getNumCols() == 2);
  CHECK(mSub->getNumRows() == numRowsOrig);

  // Sub-matrix shares data array with original.
  CHECK(mSub->getValues() == mOrig->getValues());
  CHECK(mSub->getRowIdxs() == mOrig->getRowIdxs());

  size_t * columnOffsetsOrig = mOrig->getColumnOffsets();
  size_t * columnOffsetsSub = mSub->getColumnOffsets();

  CHECK((columnOffsetsSub >= columnOffsetsOrig && columnOffsetsSub <= columnOffsetsOrig + numColsOrig));
  columnOffsetsOrig[3] = 25;
  CHECK(columnOffsetsSub[0] == 25);

  SECTION("Freeing the original matrix first is fine") {
      DataObjectFactory::destroy(mOrig);
      DataObjectFactory::destroy(mSub);
  }
  SECTION("Freeing the sub-matrix first is fine") {
      DataObjectFactory::destroy(mSub);
      DataObjectFactory::destroy(mOrig);
  }

}
