/*
 * copyright 2021 The DAPHNE Consortium
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

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/DiagVector.h>


#include <tags.h>
#include <catch.hpp>
#include <cstdint>

#define DATA_TYPES CSRMatrix, DenseMatrix
#define VALUE_TYPES  int32_t, double  

template<class DT>
void checkDiagVector(const DT * arg, DenseMatrix<typename DT::VT> *& res, DenseMatrix<typename DT::VT> * expectedMatrix) {
    diagVector<DenseMatrix<typename DT::VT>, DT>(res, arg, nullptr);
    CHECK(*res == *expectedMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("DiagVector-normal", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)

    using DT = TestType;
    auto inputMatrix = genGivenVals<DT>(3, {
        3,0,0,
        0,2,0,
        0,0,1,
    });
    auto* expectedMatrix = genGivenVals<DenseMatrix<typename DT::VT>>(3, {3, 2, 1});
    checkDiagVector(inputMatrix, expectedMatrix, expectedMatrix);
    DataObjectFactory::destroy(expectedMatrix);
    DataObjectFactory::destroy(inputMatrix);
}


TEMPLATE_PRODUCT_TEST_CASE("DiagVector-mixed-diagonal", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)

    using DT = TestType;
    auto inputMatrix = genGivenVals<DT>(3, {
        1,0,0,
        0,0,0,
        0,0,1,
    });
    auto* expectedMatrix = genGivenVals<DenseMatrix<typename DT::VT>>(3, {1, 0, 1});
    DenseMatrix<typename DT::VT> * res=nullptr;
    checkDiagVector(inputMatrix, res, expectedMatrix);
    DataObjectFactory::destroy(expectedMatrix);
    DataObjectFactory::destroy(inputMatrix);
    DataObjectFactory::destroy(res);
}


TEMPLATE_PRODUCT_TEST_CASE("DiagVector-null", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){ // NOLINT(cert-err58-cpp)

    using DT = TestType;
    auto inputMatrix = genGivenVals<DT>(3, {
        3,0,0,
        0,2,0,
        0,0,1,
    });
    auto* expectedMatrix = genGivenVals<DenseMatrix<typename DT::VT>>(3, {3, 2, 1});
    DenseMatrix<typename DT::VT> * res=nullptr;
    checkDiagVector(inputMatrix, res, expectedMatrix);
    DataObjectFactory::destroy(expectedMatrix);
    DataObjectFactory::destroy(inputMatrix);
    DataObjectFactory::destroy(res);
}
