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


 TEMPLATE_TEST_CASE("MCSCMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES){

   //using ValueType = uint32_t;
   using ValueType = TestType;
   //std::cout << typeid(ValueType).name() << std::endl;


   const size_t numRows = 4;
   const size_t numCols = 6;
   const size_t maxNumNonZeros = 8;

   MCSRMatrix<ValueType> * m = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);

   //size_t * rowSize = m -> getAllocatedRowSizes();
   /*std::cout << "rowSize: "<< rowSize << '\n';
   ValueType * row = m -> getValues(0);*/




   /*for(size_t i = 0; i<rowSize; i++){
     std::cout << columns[i] << ", ";
   }
   std::cout << "" << '\n';*/


   /*ValueType result = m -> get(0,2);
   std::cout << "result: "<< result << '\n';*/




   /*m->set(0,0,0);
   m->set(0,3,45);
   ValueType first = m->getValues(0)[0];
   ValueType sec = m->getValues(0)[1];
   //std::cout << "setted value-1: "<< static_cast<int>(m->getValues(0)[0]) << '\n';
   //std::cout << "setted value-2: "<< static_cast<int>(m->getValues(0)[1]) << '\n';

   std::cout << "setted value-1: "<< first << '\n';
   std::cout << "setted value-2: "<< sec << '\n';

   size_t * columns = m -> getColIdxs(0);

   for(size_t i = 0; i<rowSize; i++){
     std::cout << columns[i] << ", ";
   }
   std::cout << "" << '\n';

   size_t nonZeros = m-> getNumNonZeros(0);
   std::cout << "Number of non-zeros: "<< nonZeros << '\n';*/

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
   m -> append(2,5,75);
   //Sixth column
   m -> append(3,5,80);

   /*ValueType * values = m -> getValues(0);
   CHECK(values[0] == 10);

   size_t * sizes = m -> getAllNumNonZeros();

   std::cout << sizes[0] << '\n';
   std::cout << sizes[1] << '\n';
   std::cout << sizes[2] << '\n';
   std::cout << sizes[3] << '\n';*/

   MCSRMatrix<ValueType> * view = m ->sliceRow(1,3);
   //ValueType * viewVals1 = view -> getValues(0);
   //ValueType * viewVals2 = view -> getValues(1);

   //std::cout << viewVals1[0] << '\n';
   std::cout << "******************" << '\n';
   m -> print(std::cout);
   std::cout << "******************" << '\n';
   view -> print(std::cout);
   std::cout << "******************" << '\n';







   DataObjectFactory::destroy(m);



 }
