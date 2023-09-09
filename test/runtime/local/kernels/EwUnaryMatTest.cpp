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


#include <runtime/local/context/DaphneContext.h>
#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/MCSRMatrix.h>
//#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
//#include <runtime/local/kernels/EwUnarySca.h>
#include <runtime/local/kernels/EwUnaryMat.h>
//#include <runtime/local/kernels/EwUnarySca.h>



#include <cassert>
#include <cstddef>
#include <tags.h>
#include <catch.hpp>
#include <cstdint>
#include <typeinfo>
#include <iostream>



TEMPLATE_TEST_CASE("Element-wise unary with MCSR and SQRT", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros = 8;

  MCSRMatrix<ValueType> * sourceMatrix = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;  // Initialize this as necessary.
  //DCTX context = new DaphneContext(userConfig);

  DaphneContext* context = new DaphneContext(userConfig);


  //Append source matrix
  //First row
  sourceMatrix -> append(0,0,4);
  sourceMatrix -> append(0,1,4);
  //Second row
  sourceMatrix -> append(1,1,4);
  sourceMatrix -> append(1,3,4);
  //Third column
  sourceMatrix -> append(2,2,4);
  sourceMatrix -> append(2,3,4);
  sourceMatrix -> append(2,4,4);
  //Fourth row
  sourceMatrix -> append(3,5,4);

  EwUnaryMat<MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(UnaryOpCode::SQRT, resultMatrix, sourceMatrix, context);

  CHECK(resultMatrix -> get(0,0) == 2);
  CHECK(resultMatrix -> get(0,1) == 2);
  CHECK(resultMatrix -> get(1,1) == 2);
  CHECK(resultMatrix -> get(1,3) == 2);
  CHECK(resultMatrix -> get(2,2) == 2);
  CHECK(resultMatrix -> get(2,3) == 2);
  CHECK(resultMatrix -> get(2,4) == 2);
  CHECK(resultMatrix -> get(3,5) == 2);

  DataObjectFactory::destroy(sourceMatrix);
  DataObjectFactory::destroy(resultMatrix);


}




TEMPLATE_TEST_CASE("Element-wise unary with MCSR and ABS", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = int32_t;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros = 8;

  MCSRMatrix<ValueType> * sourceMatrix = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;  // Initialize this as necessary.
  //DCTX context = new DaphneContext(userConfig);

  DaphneContext* context = new DaphneContext(userConfig);


  //Append source matrix
  //First row
  sourceMatrix -> append(0,0,-4);
  sourceMatrix -> append(0,1,4);
  //Second row
  sourceMatrix -> append(1,1,-4);
  sourceMatrix -> append(1,3,4);
  //Third column
  sourceMatrix -> append(2,2,-4);
  sourceMatrix -> append(2,3,4);
  sourceMatrix -> append(2,4,-4);
  //Fourth row
  sourceMatrix -> append(3,5,4);

  EwUnaryMat<MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(UnaryOpCode::ABS, resultMatrix, sourceMatrix, context);

  CHECK(resultMatrix -> get(0,0) == 4);
  CHECK(resultMatrix -> get(0,1) == 4);
  CHECK(resultMatrix -> get(1,1) == 4);
  CHECK(resultMatrix -> get(1,3) == 4);
  CHECK(resultMatrix -> get(2,2) == 4);
  CHECK(resultMatrix -> get(2,3) == 4);
  CHECK(resultMatrix -> get(2,4) == 4);
  CHECK(resultMatrix -> get(3,5) == 4);

  DataObjectFactory::destroy(sourceMatrix);
  DataObjectFactory::destroy(resultMatrix);


}


TEMPLATE_TEST_CASE("Element-wise unary with MCSR and SIGN", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = int32_t;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros = 8;

  MCSRMatrix<ValueType> * sourceMatrix = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;  // Initialize this as necessary.
  //DCTX context = new DaphneContext(userConfig);

  DaphneContext* context = new DaphneContext(userConfig);


  //Append source matrix
  //First row
  sourceMatrix -> append(0,0,-4);
  sourceMatrix -> append(0,1,4);
  //Second row
  sourceMatrix -> append(1,1,-4);
  sourceMatrix -> append(1,3,4);
  //Third column
  sourceMatrix -> append(2,2,-4);
  sourceMatrix -> append(2,3,4);
  sourceMatrix -> append(2,4,-4);
  //Fourth row
  sourceMatrix -> append(3,5,4);

  EwUnaryMat<MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(UnaryOpCode::SIGN, resultMatrix, sourceMatrix, context);

  CHECK(resultMatrix -> get(0,0) == -1);
  CHECK(resultMatrix -> get(0,1) == 1);
  CHECK(resultMatrix -> get(1,1) == -1);
  CHECK(resultMatrix -> get(1,3) == 1);
  CHECK(resultMatrix -> get(2,2) == -1);
  CHECK(resultMatrix -> get(2,3) == 1);
  CHECK(resultMatrix -> get(2,4) == -1);
  CHECK(resultMatrix -> get(3,5) == 1);

  DataObjectFactory::destroy(sourceMatrix);
  DataObjectFactory::destroy(resultMatrix);


}
