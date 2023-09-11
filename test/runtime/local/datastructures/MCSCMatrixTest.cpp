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

 #include <runtime/local/datastructures/MCSCMatrix.h>
 #include <runtime/local/datastructures/DataObjectFactory.h>
 #include <runtime/local/datastructures/ValueTypeUtils.h>

 #include <tags.h>
 #include <catch.hpp>
 #include <cstdint>
 #include <typeinfo>
 #include <iostream>


 TEMPLATE_TEST_CASE("MCSCMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 8;

   MCSCMatrix<ValueType> * m = DataObjectFactory::create<MCSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   CHECK(m->getNumRows() == 4);
   CHECK(m->getNumCols() == 6);
   CHECK(m->getMaxNumNonZeros() == 8);

   DataObjectFactory::destroy(m);

 }



 TEMPLATE_TEST_CASE("MCSCMatrix appends and updates values correctly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 9;

   MCSCMatrix<ValueType> * m = DataObjectFactory::create<MCSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   m -> append(0,0,10);
   m -> append(0,1,20);
   m -> append(1,1,30);
   m -> append(2,2,50);
   m -> append(1,3,40);
   m -> append(2,3,60);
   m -> append(2,4,70);
   m -> append(3,5,80);

   ValueType * firstCol  = m -> getValues(0);
   ValueType * secondCol = m -> getValues(1);
   ValueType * thirdCol  = m -> getValues(2);
   ValueType * fourthCol = m -> getValues(3);
   ValueType * fithCol = m -> getValues(4);
   ValueType * sixthCol = m -> getValues(5);

   //Check that values were appended correctly
   CHECK(firstCol[0]  == 10);
   CHECK(secondCol[0]  == 20);
   CHECK(secondCol[1] == 30);
   CHECK(thirdCol[0] == 50);
   CHECK(fourthCol[0]  == 40);
   CHECK(fourthCol[1]  == 60);
   CHECK(fithCol[0]  == 70);
   CHECK(sixthCol[0] == 80);

   m -> set(0,0,5);
   CHECK(firstCol[0]  == 5);

   m -> set(1,0,15);
   CHECK(firstCol[1]  == 15);

   DataObjectFactory::destroy(m);

 }



 TEMPLATE_TEST_CASE("MCSCMatrix sub-matrix works properly", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   using ValueType = TestType;

   const size_t numRowsOrig = 4;
   const size_t numColsOrig = 6;
   const size_t maxNumNonZeros = 8;
   size_t colLowerIncl = 1;
   size_t colUpperExcl = 3;

   MCSCMatrix<ValueType> * mOrig = DataObjectFactory::create<MCSCMatrix<ValueType>>(numRowsOrig, numColsOrig, maxNumNonZeros, true);
   MCSCMatrix<ValueType> * mSub = mOrig -> sliceCol(colLowerIncl, colUpperExcl);

   //Append values of original matrix
   mOrig -> append(0,0,10);
   mOrig -> append(0,1,20);
   mOrig -> append(1,1,30);
   mOrig -> append(2,2,50);
   mOrig -> append(1,3,40);
   mOrig -> append(2,3,60);
   mOrig -> append(2,4,70);
   mOrig -> append(3,5,80);

   //Sub-matrix dimensions are as expected.
   CHECK(mSub->getNumRows() == numRowsOrig);
   CHECK(mSub->getNumCols() == 2);

   ValueType * firstColSub = mSub -> getValues(0);
   ValueType * secondColSub = mSub -> getValues(1);
   ValueType * secondColOrig = mOrig -> getValues(1);
   ValueType * thirdColOrig = mOrig -> getValues(2);

   //Compare values
   CHECK(firstColSub[0] == secondColOrig[0]);
   CHECK(firstColSub[1] == secondColOrig[1]);
   CHECK(secondColSub[0] == thirdColOrig[0]);


   //Sub-matrix and original matrix share updates
   //Replace 30 at position (1,1) with 35
   mOrig -> set(1,1,35);
   CHECK(mSub -> get(1,0) == 35);
   CHECK(firstColSub[1] == 35);
   CHECK(mOrig->getNumNonZeros(1) == mSub->getNumNonZeros(0));

   //Replace 35 at position (1,1) with 0 in the original matrix
   mOrig -> set(1,1,0);
   CHECK(mSub -> get(1,0) == 0);
   CHECK(mOrig->getNumNonZeros(1) == mSub->getNumNonZeros(0));



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
