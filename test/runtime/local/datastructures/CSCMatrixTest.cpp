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


TEMPLATE_TEST_CASE("CSCMatrix", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
     // No assertions in this test case. We just want to see if it runs without
     // crashing.

     using ValueType = TestType;

      const size_t numRows = 4;
      const size_t numCols = 6;
      const size_t maxNumNonZeros = 8;

     CSCMatrix<ValueType> * m = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);
     //CSCMatrix<ValueType> * m = new CSCMatrix<ValueType>(numRows, numCols, numNonZeros);

     size_t numRowsNew = m -> getNumRows();
     size_t numColsNew = m -> getNumCols();
     size_t maxNumNonZerosNew = m -> getMaxNumNonZeros();
     size_t numNonZeros = m -> getNumNonZeros();
     std::cout << "Number of rows: " << numRowsNew << '\n';
     std::cout << "Number of columns: " << numColsNew << '\n';
     std::cout << "Maximum number of non-zeros: " << maxNumNonZerosNew <<'\n';
     std::cout << "Number of non-zeros: " << numNonZeros <<'\n';










     DataObjectFactory::destroy(m);
 }
