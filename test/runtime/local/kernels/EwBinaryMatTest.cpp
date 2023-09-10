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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/MCSRMatrix.h>
#include <runtime/local/context/DaphneContext.h>
#include <api/cli/DaphneUserConfig.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/EwBinaryMat.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

#define TEST_NAME(opName) "EwBinaryMat (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES double, uint32_t

template<class DT>
void checkEwBinaryMat(BinaryOpCode opCode, const DT * lhs, const DT * rhs, const DT * exp) {
    DT * res = nullptr;
    ewBinaryMat<DT, DT, DT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
}

template<class SparseDT, class DT>
void checkSparseDenseEwBinaryMat(BinaryOpCode opCode, const SparseDT * lhs, const DT * rhs, const SparseDT * exp) {
    SparseDT * res = nullptr;
    ewBinaryMat<SparseDT, SparseDT, DT>(opCode, res, lhs, rhs, nullptr);
    CHECK(*res == *exp);
}

// ****************************************************************************
// Arithmetic
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto m0 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            1, 2, 3, 1, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 1, 0, 2,
    });
    auto m3 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            1, 3, 3, 3, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 1, 0, 2,
    });

    checkEwBinaryMat(BinaryOpCode::ADD, m0, m0, m0);
    checkEwBinaryMat(BinaryOpCode::ADD, m1, m0, m1);
    checkEwBinaryMat(BinaryOpCode::ADD, m1, m2, m3);

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto m0 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(4, {
            1, 2, 0, 0, 1, 3,
            0, 1, 0, 2, 0, 3,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            1, 2, 3, 1, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 3, 1, 0, 2,
    });
    auto m3 = genGivenVals<DT>(4, {
            0, 0, 0, 0, 0, 0,
            0, 2, 0, 2, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
    });

    checkEwBinaryMat(BinaryOpCode::MUL, m0, m0, m0);
    checkEwBinaryMat(BinaryOpCode::MUL, m1, m0, m0);
    checkEwBinaryMat(BinaryOpCode::MUL, m1, m2, m3);

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_TEST_CASE(TEST_NAME("mul_sparse_dense"), TAG_KERNELS, VALUE_TYPES) {
    // TODO: all Dense - CSR combinations
    using VT = TestType;
    using SparseDT = CSRMatrix<VT>;
    using DT = DenseMatrix<VT>;

    auto m0 = genGivenVals<SparseDT>(4, {
        0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    });

    auto m1 = genGivenVals<DT>(4, {
        1, 2, 0, 0, 1, 3,
        0, 1, 0, 2, 0, 3,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    });
    auto m2 = genGivenVals<DT>(4, {
        3, 0, 3, 3, 3, 3,
        1, 2, 3, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 2, 3, 1, 3, 2,
    });
    auto m3 = genGivenVals<DT>(4, {
        0, 1, 1, 0, 0, 0,
        0, 2, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    });
    auto exp0 = genGivenVals<SparseDT>(4, {
        0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    });
    auto exp1 = genGivenVals<SparseDT>(4, {
        0, 0, 3, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    });

    checkSparseDenseEwBinaryMat(BinaryOpCode::MUL, m0, m1, exp0);
    checkSparseDenseEwBinaryMat(BinaryOpCode::MUL, m0, m2, exp1);
    checkSparseDenseEwBinaryMat(BinaryOpCode::MUL, m0, m3, m0);

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
    DataObjectFactory::destroy(exp0);
    DataObjectFactory::destroy(exp1);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m0 = genGivenVals<DT>(2, {
            0, 0, 0,
            0, 0, 0,
    });
    auto m1 = genGivenVals<DT>(2, {
            1, 2, 4,
            6, 8, 9,
    });
    auto m2 = genGivenVals<DT>(2, {
            1, 2, 2,
            2, 4, 3,
    });
    auto m3 = genGivenVals<DT>(2, {
            1, 1, 2,
            3, 2, 3,
    });

    checkEwBinaryMat(BinaryOpCode::DIV, m0, m1, m0);
    checkEwBinaryMat(BinaryOpCode::DIV, m1, m2, m3);

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

// ****************************************************************************
// Comparisons
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 3,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {1, 0, 1,  1, 0, 0,});

    checkEwBinaryMat(BinaryOpCode::EQ, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 3,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {0, 1, 0,  0, 1, 1,});

    checkEwBinaryMat(BinaryOpCode::NEQ, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {0, 0, 1,  0, 0, 1,});

    checkEwBinaryMat(BinaryOpCode::LT, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {1, 0, 1,  1, 0, 1,});

    checkEwBinaryMat(BinaryOpCode::LE, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {0, 1, 0,  0, 1, 0,});

    checkEwBinaryMat(BinaryOpCode::GT, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {1, 1, 0,  1, 1, 0,});

    checkEwBinaryMat(BinaryOpCode::GE, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

// ****************************************************************************
// Min/max
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {1, 0, 3,  4, 4, 6,});

    checkEwBinaryMat(BinaryOpCode::MIN, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;

    auto m1 = genGivenVals<DT>(2, {1, 2, 3,  4, 5, 6,});
    auto m2 = genGivenVals<DT>(2, {1, 0, 4,  4, 4, 9,});
    auto m3 = genGivenVals<DT>(2, {1, 2, 4,  4, 5, 9,});

    checkEwBinaryMat(BinaryOpCode::MAX, m1, m2, m3);

    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);
    DataObjectFactory::destroy(m3);
}

// ****************************************************************************
// Logical
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto m1 = genGivenVals<DT>(1, {0, 0, 1, 1, 0, 2, 2,     0, VT(-2), VT(-2)});
    auto m2 = genGivenVals<DT>(1, {0, 1, 0, 1, 2, 0, 2, VT(-2),    0 , VT(-2)});
    auto m3 = genGivenVals<DT>(1, {0, 0, 0, 1, 0, 0, 1,     0 ,    0 ,     1 });

    checkEwBinaryMat(BinaryOpCode::AND, m1, m2, m3);

    DataObjectFactory::destroy(m1, m2, m3);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    auto m1 = genGivenVals<DT>(1, {0, 0, 1, 1, 0, 2, 2,     0 , VT(-2), VT(-2)});
    auto m2 = genGivenVals<DT>(1, {0, 1, 0, 1, 2, 0, 2, VT(-2),     0 , VT(-2)});
    auto m3 = genGivenVals<DT>(1, {0, 1, 1, 1, 1, 1, 1,     1,      1 ,     1 });

    checkEwBinaryMat(BinaryOpCode::OR, m1, m2, m3);

    DataObjectFactory::destroy(m1, m2, m3);
}

// ****************************************************************************
// Invalid op-code
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    DT * res = nullptr;
    auto m = genGivenVals<DT>(1, {1});
    CHECK_THROWS(ewBinaryMat<DT, DT, DT>(static_cast<BinaryOpCode>(999), res, m, m, nullptr));
}





// ****************************************************************************
// MCSR Matrix
// ****************************************************************************



TEMPLATE_TEST_CASE("Element-wise binary with MCSR and ADD", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros1 = 8;
  const size_t maxNumNonZeros2 = 5;

  MCSRMatrix<ValueType> * lhs = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros1, true);
  MCSRMatrix<ValueType> * rhs = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros2, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;
  DaphneContext* context = new DaphneContext(userConfig);

  //Append source matrix
  //First row
  lhs -> append(0,0,10);    rhs -> append(0,2,20);
  lhs -> append(0,1,20);    rhs -> append(0,5,10);
  //Second row
  lhs -> append(1,1,30);    rhs -> append(1,1,5);
  lhs -> append(1,3,40);    rhs -> append(1,2,5);
  //Third column
  lhs -> append(2,2,50);
  lhs -> append(2,3,60);
  lhs -> append(2,4,70);
  //Fourth row
  lhs -> append(3,5,80);    rhs -> append(3,2,10);

  EwBinaryMat<MCSRMatrix<ValueType>,MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(BinaryOpCode::ADD, resultMatrix, lhs, rhs, context);

  CHECK(resultMatrix -> get(0,0) == 10);
  CHECK(resultMatrix -> get(0,1) == 20);
  CHECK(resultMatrix -> get(0,2) == 20);
  CHECK(resultMatrix -> get(0,5) == 10);
  CHECK(resultMatrix -> get(1,1) == 35);
  CHECK(resultMatrix -> get(1,2) == 5);
  CHECK(resultMatrix -> get(1,3) == 40);
  CHECK(resultMatrix -> get(2,2) == 50);
  CHECK(resultMatrix -> get(2,3) == 60);
  CHECK(resultMatrix -> get(2,4) == 70);
  CHECK(resultMatrix -> get(3,2) == 10);
  CHECK(resultMatrix -> get(3,5) == 80);

  DataObjectFactory::destroy(lhs);
  DataObjectFactory::destroy(rhs);
  DataObjectFactory::destroy(resultMatrix);


}


TEMPLATE_TEST_CASE("Element-wise binary with MCSR and MUL", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros1 = 8;
  const size_t maxNumNonZeros2 = 5;

  MCSRMatrix<ValueType> * lhs = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros1, true);
  MCSRMatrix<ValueType> * rhs = DataObjectFactory::create<MCSRMatrix<ValueType>>(numRows, numCols, maxNumNonZeros2, true);
  MCSRMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;
  DaphneContext* context = new DaphneContext(userConfig);

  //Append source matrix
  //First row
  lhs -> append(0,0,10);    rhs -> append(0,2,1);
  lhs -> append(0,1,20);    rhs -> append(0,5,1);
  //Second row
  lhs -> append(1,1,30);    rhs -> append(1,1,1);
  lhs -> append(1,3,40);    rhs -> append(1,2,1);
  //Third column
  lhs -> append(2,2,50);
  lhs -> append(2,3,60);    rhs -> append(2,3,1);
  lhs -> append(2,4,70);
  //Fourth row
  lhs -> append(3,5,80);    rhs -> append(3,2,1);

  EwBinaryMat<MCSRMatrix<ValueType>,MCSRMatrix<ValueType>, MCSRMatrix<ValueType>>::apply(BinaryOpCode::MUL, resultMatrix, lhs, rhs, context);

  CHECK(resultMatrix -> get(0,0) == 0);
  CHECK(resultMatrix -> get(0,1) == 0);
  CHECK(resultMatrix -> get(0,2) == 0);
  CHECK(resultMatrix -> get(0,5) == 0);
  CHECK(resultMatrix -> get(1,1) == 30);
  CHECK(resultMatrix -> get(2,3) == 60);
  CHECK(resultMatrix -> get(3,5) == 0);



  DataObjectFactory::destroy(lhs);
  DataObjectFactory::destroy(rhs);
  DataObjectFactory::destroy(resultMatrix);


}


// ****************************************************************************
// CSCMatrix
// ****************************************************************************

TEMPLATE_TEST_CASE("Element-wise binary with CSC and ADD", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros1 = 14;
  const size_t maxNumNonZeros2 = 9;

  CSCMatrix<ValueType> * lhs = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros1, true);
  CSCMatrix<ValueType> * rhs = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros2, true);
  CSCMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;
  DaphneContext* context = new DaphneContext(userConfig);

  //Append source matrices
  lhs -> prepareAppend();
  lhs -> append(0,0,1);
  lhs -> append(3,0,1);
  lhs -> append(0,1,1);
  lhs -> append(2,1,1);
  lhs -> append(3,1,1);
  lhs -> append(1,2,1);
  lhs -> append(3,2,1);
  lhs -> append(0,3,1);
  lhs -> append(2,3,1);
  lhs -> append(3,3,1);
  lhs -> append(0,4,1);
  lhs -> append(3,4,1);
  lhs -> append(2,5,1);
  lhs -> append(3,5,1);
  lhs -> finishAppend();

  rhs -> prepareAppend();
  rhs -> append(0,0,1);
  rhs -> append(1,0,1);
  rhs -> append(3,0,1);
  rhs -> append(1,1,1);
  rhs -> append(0,2,1);
  rhs -> append(3,2,1);
  rhs -> append(3,3,1);
  rhs -> append(0,4,1);
  rhs -> append(1,5,1);
  rhs -> finishAppend();


  EwBinaryMat<CSCMatrix<ValueType>,CSCMatrix<ValueType>, CSCMatrix<ValueType>>::apply(BinaryOpCode::ADD, resultMatrix, lhs, rhs, context);

  CHECK(resultMatrix -> get(0,0) == 2);
  CHECK(resultMatrix -> get(0,1) == 1);
  CHECK(resultMatrix -> get(0,2) == 1);
  CHECK(resultMatrix -> get(0,3) == 1);
  CHECK(resultMatrix -> get(0,4) == 2);
  CHECK(resultMatrix -> get(1,0) == 1);
  CHECK(resultMatrix -> get(1,1) == 1);
  CHECK(resultMatrix -> get(1,2) == 1);
  CHECK(resultMatrix -> get(1,5) == 1);
  CHECK(resultMatrix -> get(2,1) == 1);
  CHECK(resultMatrix -> get(2,3) == 1);
  CHECK(resultMatrix -> get(2,5) == 1);
  CHECK(resultMatrix -> get(3,0) == 2);
  CHECK(resultMatrix -> get(3,1) == 1);
  CHECK(resultMatrix -> get(3,2) == 2);
  CHECK(resultMatrix -> get(3,3) == 2);
  CHECK(resultMatrix -> get(3,4) == 1);
  CHECK(resultMatrix -> get(3,5) == 1);


  DataObjectFactory::destroy(lhs);
  DataObjectFactory::destroy(rhs);
  DataObjectFactory::destroy(resultMatrix);

}



TEMPLATE_TEST_CASE("Element-wise binary with CSC and MUL", TAG_KERNELS, ALL_VALUE_TYPES){

  using ValueType = TestType;

  const size_t numRows = 4;
  const size_t numCols = 6;
  const size_t maxNumNonZeros1 = 14;
  const size_t maxNumNonZeros2 = 9;

  CSCMatrix<ValueType> * lhs = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros1, true);
  CSCMatrix<ValueType> * rhs = DataObjectFactory::create<CSCMatrix<ValueType>>(numRows, numCols, maxNumNonZeros2, true);
  CSCMatrix<ValueType> * resultMatrix = nullptr;

  DaphneUserConfig userConfig;
  DaphneContext* context = new DaphneContext(userConfig);

  //Append source matrices
  lhs -> prepareAppend();
  lhs -> append(0,0,1);
  lhs -> append(3,0,1);
  lhs -> append(0,1,1);
  lhs -> append(2,1,1);
  lhs -> append(3,1,1);
  lhs -> append(1,2,1);
  lhs -> append(3,2,1);
  lhs -> append(0,3,1);
  lhs -> append(2,3,1);
  lhs -> append(3,3,1);
  lhs -> append(0,4,1);
  lhs -> append(3,4,1);
  lhs -> append(2,5,1);
  lhs -> append(3,5,1);
  lhs -> finishAppend();

  rhs -> prepareAppend();
  rhs -> append(0,0,1);
  rhs -> append(1,0,1);
  rhs -> append(3,0,1);
  rhs -> append(1,1,1);
  rhs -> append(0,2,1);
  rhs -> append(3,2,1);
  rhs -> append(3,3,1);
  rhs -> append(0,4,1);
  rhs -> append(1,5,1);
  rhs -> finishAppend();


  EwBinaryMat<CSCMatrix<ValueType>,CSCMatrix<ValueType>, CSCMatrix<ValueType>>::apply(BinaryOpCode::MUL, resultMatrix, lhs, rhs, context);

  CHECK(resultMatrix -> get(0,0) == 1);
  CHECK(resultMatrix -> get(0,1) == 0);
  CHECK(resultMatrix -> get(0,2) == 0);
  CHECK(resultMatrix -> get(0,3) == 0);
  CHECK(resultMatrix -> get(0,4) == 1);
  CHECK(resultMatrix -> get(1,0) == 0);
  CHECK(resultMatrix -> get(1,1) == 0);
  CHECK(resultMatrix -> get(1,2) == 0);
  CHECK(resultMatrix -> get(1,5) == 0);
  CHECK(resultMatrix -> get(2,1) == 0);
  CHECK(resultMatrix -> get(2,3) == 0);
  CHECK(resultMatrix -> get(2,5) == 0);
  CHECK(resultMatrix -> get(3,0) == 1);
  CHECK(resultMatrix -> get(3,1) == 0);
  CHECK(resultMatrix -> get(3,2) == 1);
  CHECK(resultMatrix -> get(3,3) == 1);
  CHECK(resultMatrix -> get(3,4) == 0);
  CHECK(resultMatrix -> get(3,5) == 0);


  DataObjectFactory::destroy(lhs);
  DataObjectFactory::destroy(rhs);
  DataObjectFactory::destroy(resultMatrix);

}
