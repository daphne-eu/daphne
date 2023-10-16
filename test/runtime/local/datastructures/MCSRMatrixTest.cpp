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

 #include <runtime/local/datastructures/MCSRMatrix.h>
 #include <runtime/local/datastructures/DataObjectFactory.h>
 #include <runtime/local/datastructures/ValueTypeUtils.h>

 #include <tags.h>
 #include <catch.hpp>
 #include <cstdint>
 #include <typeinfo>
 #include <iostream>


 TEMPLATE_TEST_CASE("MCSRMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 8;

   MCSRMatrix<ValueType> * m = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   CHECK(m->getNumRows() == 4);
   CHECK(m->getNumCols() == 6);
   CHECK(m->getMaxNumNonZeros() == 8);

   DataObjectFactory::destroy(m);

 }


 TEMPLATE_TEST_CASE("MCSRMatrix appends and updates values correctly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 10;

   MCSRMatrix<ValueType> * m = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   //This example requires MCSR to reallocate additional memory for the third row
   //First row
   m -> append(0,0,10);
   m -> append(0,1,20);
   //Second row
   m -> append(1,1,30);
   m -> append(1,3,40);
   //Third column
   m -> append(2,2,50);
   m -> append(2,3,60);
   m -> append(2,4,70);
   m -> append(2,5,75);
   //Fourth row
   m -> append(3,5,80);

   //Get rows by rowIdx
   ValueType * firstRow  = m -> getValues(0);
   ValueType * secondRow = m -> getValues(1);
   ValueType * thirdRow  = m -> getValues(2);
   ValueType * fourthRow = m -> getValues(3);

   //Check that values were appended correctly
   CHECK(firstRow[0]  == 10);
   CHECK(firstRow[1]  == 20);
   CHECK(secondRow[0] == 30);
   CHECK(secondRow[1] == 40);
   CHECK(thirdRow[0]  == 50);
   CHECK(thirdRow[1]  == 60);
   CHECK(thirdRow[2]  == 70);
   CHECK(thirdRow[3]  == 75);
   CHECK(fourthRow[0] == 80);

   //Replace a non-zero value with another
   //Replace 10 with 15
   //Number of non-zeros in that row must not chnage
   size_t numNonZerosBefore1 = m -> getNumNonZeros(0);
   m -> set(0,0,15);
   CHECK(m -> get(0,0) == 15);
   size_t numNonZerosAfter1 = m -> getNumNonZeros(0);
   CHECK(numNonZerosBefore1 == numNonZerosAfter1);

   //Set non-zero value at a zero position
   //Replace 0 at position (0,2) with 5
   //Number of non-zeros in that row must chnage (increase)
   size_t numNonZerosBefore2 = m -> getNumNonZeros(0);
   m -> set(0,2,5);
   CHECK(m -> get(0,2) == 5);
   size_t numNonZerosAfter2 = m -> getNumNonZeros(0);
   CHECK(numNonZerosBefore2 < numNonZerosAfter2);

   //Set zero value at a non-zero position
   //Replace 50 at position (2,2) with 0
   //Since we dont store zeros, the first element in that row must become 60
   //Number of non-zeros in that row must chnage (decrease)
   size_t numNonZerosBefore3 = m -> getNumNonZeros(2);
   m -> set(2,2,0);
   CHECK(m -> get(2,2) == 0);
   CHECK(thirdRow[0] == 60);
   size_t numNonZerosAfter3 = m -> getNumNonZeros(2);
   CHECK(numNonZerosBefore3 > numNonZerosAfter3);

   //Set zero value at a zero position
   //Nothing should change since we dont store zeros
   //Replace 0 at position (3,0) with 0
   size_t numNonZerosBefore4 = m -> getNumNonZeros(3);
   m -> set(3,0,0);
   CHECK(m -> get(3,0) == 0);
   CHECK(fourthRow[0] == 80);
   size_t numNonZerosAfter4 = m -> getNumNonZeros(3);
   CHECK(numNonZerosBefore4 == numNonZerosAfter4);

   DataObjectFactory::destroy(m);

 }


 TEMPLATE_TEST_CASE("MCSRMatrix sub-matrix works properly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRowsOrig = 4;
   const size_t numColsOrig = 6;
   const size_t maxNumNonZeros = 8;
   size_t rowLowerIncl = 1;
   size_t rowUpperExcl = 3;

   MCSRMatrix<ValueType> * mOrig = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRowsOrig, numColsOrig, maxNumNonZeros, true);
   MCSRMatrix<ValueType> * mSub = mOrig -> sliceRow(rowLowerIncl, rowUpperExcl);

   //Append values of original matrix
   mOrig -> append(0,0,10);
   mOrig -> append(0,1,20);
   //Second row
   mOrig -> append(1,1,30);
   mOrig -> append(1,3,40);
   //Third column
   mOrig -> append(2,2,50);
   mOrig -> append(2,3,60);
   mOrig -> append(2,4,70);
   //Fourth row
   mOrig -> append(3,5,80);

   //Sub-matrix dimensions are as expected.
   CHECK(mSub->getNumCols() == numColsOrig);
   CHECK(mSub->getNumRows() == 2);

   ValueType * firstRowSub = mSub -> getValues(0);
   ValueType * secondRowSub = mSub -> getValues(1);
   ValueType * secondRowOrig = mOrig -> getValues(1);
   ValueType * thirdRowOrig = mOrig -> getValues(2);

   //Compare values
   CHECK(firstRowSub[0] == secondRowOrig[0]);
   CHECK(firstRowSub[1] == secondRowOrig[1]);
   CHECK(secondRowSub[0] == thirdRowOrig[0]);
   CHECK(secondRowSub[1] == thirdRowOrig[1]);
   CHECK(secondRowSub[2] == thirdRowOrig[2]);

   //Sub-matrix and original matrix share updates
   //Replace 30 at position (1,1) with 35
   mOrig -> set(1,1,35);
   CHECK(mSub -> get(0,1) == 35);
   CHECK(firstRowSub[0] == 35);
   CHECK(mOrig->getNumNonZeros(1) == mSub->getNumNonZeros(0));

   //Replace 35 at position (1,1) with 0 in the original matrix
   //New first element is 40
   mOrig -> set(1,1,0);
   CHECK(mSub -> get(0,1) == 0);
   CHECK(firstRowSub[0] == 40);
   CHECK(mOrig->getNumNonZeros(1) == mSub->getNumNonZeros(0));

   //Check number of non-zeros for second row of sub-matrix
   CHECK(mOrig->getNumNonZeros(2) == mSub->getNumNonZeros(1));

   //Memory is only freed after all matrices free shared memory
   SECTION("Freeing the original matrix first is fine") {
       DataObjectFactory::destroy(mOrig);
       DataObjectFactory::destroy(mSub);
   }
   SECTION("Freeing the sub-matrix first is fine") {
       DataObjectFactory::destroy(mSub);
       DataObjectFactory::destroy(mOrig);
   }


 }
