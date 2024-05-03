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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/Replace.h>
#include <runtime/local/kernels/CheckEq.h>


#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES double, float, uint32_t, uint64_t, int32_t, int64_t
#define VALUE_TYPES_SPECIAL_CASE double

template<class DT>
void checkReplace(DT*& outputMatrix,  const DT* inputMatrix,typename DT::VT pattern, typename DT::VT replacement,  DT* expected){
    replace<DT, DT, typename DT::VT>(outputMatrix, inputMatrix, pattern, replacement, nullptr);
    CHECK(*outputMatrix == *expected);
}

TEMPLATE_PRODUCT_TEST_CASE("Replace", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)){
    using DT = TestType;
    //inplace updates

    auto initMatrix = genGivenVals<DT>(4, {
        1, 2, 3, 7, 7, 7,
        7, 1, 2, 3, 7, 7,
        7, 7, 1, 2, 3, 7,
        7, 7, 7, 1, 2, 3,
    });

    auto testMatrix1 = genGivenVals<DT>(4, {
        7, 2, 3, 7, 7, 7,
        7, 7, 2, 3, 7, 7,
        7, 7, 7, 2, 3, 7,
        7, 7, 7, 7, 2, 3,
    });

    auto testMatrix2 = genGivenVals<DT>(4, {
        7, 7, 3, 7, 7, 7,
        7, 7, 7, 3, 7, 7,
        7, 7, 7, 7, 3, 7,
        7, 7, 7, 7, 7, 3,
    });

    checkReplace(initMatrix, initMatrix, 1, 7, testMatrix1);
    //should do nothing because there is no ones
    checkReplace(initMatrix, initMatrix, 1, 7, testMatrix1);
    //target=2;
    checkReplace(initMatrix, initMatrix, 2, 7, testMatrix2);

    // update in a new copy
    auto testMatrix3 = genGivenVals<DT>(4, {
        7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7,
    });

    auto testMatrix4 = genGivenVals<DT>(4, {
        7, 7, 10, 7, 7, 7,
        7, 7, 7, 10, 7, 7,
        7, 7, 7, 7, 10, 7,
        7, 7, 7, 7, 7, 10,
    });

    DT * outputMatrix=nullptr;
    checkReplace(outputMatrix, initMatrix, 3, 7, testMatrix3);

    checkReplace(initMatrix, initMatrix, 3, 10, testMatrix4);
    //this test case should act as a copy
    DT * outputMatrix2=nullptr;

    checkReplace(outputMatrix2, initMatrix,  3, 3, testMatrix4);
    DataObjectFactory::destroy(initMatrix);
    DataObjectFactory::destroy(testMatrix1);
    DataObjectFactory::destroy(testMatrix2);
    DataObjectFactory::destroy(testMatrix3);
    DataObjectFactory::destroy(testMatrix4);
    DataObjectFactory::destroy(outputMatrix);
    DataObjectFactory::destroy(outputMatrix2);
}

TEMPLATE_PRODUCT_TEST_CASE("Replace-nan", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES_SPECIAL_CASE)){
    using DT = TestType;
    //inplace updates

    auto initMatrix = genGivenVals<DT>(4, {
        1, 2, 3, 7, 7, 7,
        7, 1, 2, 3, 7, 7,
        7, 7, 1, 2, 3, 7,
        7, 7, 7, 1, 2, 3,
    });
    auto testMatrix1 = genGivenVals<DT>(4, {
        1000, 2, 3, 7, 7, 7,
        7, 1, 2, 3, 7, 7,
        7, 7, 1, 2, 3, 7,
        7, 7, 7, 1, 2, 3,
    });

    initMatrix->set(0, 0, nan(""));
    checkReplace(initMatrix, initMatrix ,nan(""), 1000, testMatrix1);
    DataObjectFactory::destroy(initMatrix);
    DataObjectFactory::destroy(testMatrix1);

}