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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/MCSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Transpose.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

template<class DT>
void checkTranspose(const DT * arg, const DT * exp) {
    DT * res = nullptr;
    transpose<DT, DT>(res, arg, nullptr);
    CHECK(*res == *exp);
}

TEMPLATE_PRODUCT_TEST_CASE("Transpose", TAG_KERNELS, (DenseMatrix, CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    DT * m = nullptr;
    DT * mt = nullptr;

    SECTION("fully populated matrix") {
        m = genGivenVals<DT>(3, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
        });
        mt = genGivenVals<DT>(4, {
            1, 5,  9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12,
        });
    }
    SECTION("sparse matrix") {
        m = genGivenVals<DT>(5, {
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 0, 0, 0,
            0, 0, 0, 0, 4, 0,
            0, 0, 0, 0, 0, 0,
            5, 0, 0, 0, 6, 0,
        });
        mt = genGivenVals<DT>(6, {
            0, 0, 0, 0, 5,
            0, 0, 0, 0, 0,
            0, 3, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 4, 0, 6,
            0, 0, 0, 0, 0,
        });
    }
    SECTION("empty matrix") {
        m = genGivenVals<DT>(3, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        });
        mt = genGivenVals<DT>(4, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        });
    }

    checkTranspose(m, mt);

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(mt);
}


TEMPLATE_TEST_CASE("Transpose for MCSR works", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros = 8;

  MCSRMatrix<ValueType> * sourceMatrix = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;
  DaphneContext* context = new DaphneContext(userConfig);


  //Append source matrix
  //First row
  sourceMatrix -> append(0,0,10);
  sourceMatrix -> append(0,1,20);
  //Second row
  sourceMatrix -> append(1,1,30);
  sourceMatrix -> append(1,3,40);
  //Third column
  sourceMatrix -> append(2,2,50);
  sourceMatrix -> append(2,3,60);
  sourceMatrix -> append(2,4,70);
  //Fourth row
  sourceMatrix -> append(3,5,80);

  Transpose<MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(resultMatrix, sourceMatrix, context);

  CHECK(resultMatrix->getNumRows() == sourceMatrix->getNumCols());
  CHECK(resultMatrix->getNumCols() == sourceMatrix->getNumRows());

  CHECK(resultMatrix -> get(0,0) == 10);
  CHECK(resultMatrix -> get(1,0) == 20);
  CHECK(resultMatrix -> get(1,1) == 30);
  CHECK(resultMatrix -> get(2,2) == 50);
  CHECK(resultMatrix -> get(3,1) == 40);
  CHECK(resultMatrix -> get(3,2) == 60);
  CHECK(resultMatrix -> get(4,2) == 70);
  CHECK(resultMatrix -> get(5,3) == 80);


  DataObjectFactory::destroy(sourceMatrix);
  DataObjectFactory::destroy(resultMatrix);


}
