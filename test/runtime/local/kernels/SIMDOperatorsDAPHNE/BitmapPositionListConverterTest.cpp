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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/BitmapPositionListConverter.h>

#include <mlir/IR/Location.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

TEST_CASE("BitmapPositionListConverter: Test case 1", TAG_KERNELS) {
    /// Data generation
    auto pos_list = genGivenVals<DenseMatrix<size_t>>(10, { 0,  0,   0, 0, 1, 0, 1, 1, 0, 0});
    
    DenseMatrix<size_t> * expectedResult;
    /// create expected result set
    {
        std::vector<size_t> er_col0_val;
        for (size_t outerLoop = 0; outerLoop < pos_list->getNumRows(); outerLoop++) {
            if(pos_list->get(outerLoop, 0) == 1) {
                    er_col0_val.push_back(outerLoop);
            }
        }
        size_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<size_t>>(size, er_col0_val);
    }
    
    /// test execution
    DenseMatrix<size_t> * resultMatrix = nullptr;

    bitmapPositionListConverter(resultMatrix, pos_list, nullptr);    
    
    /// test if result matches expected result
    CHECK(checkEq<DenseMatrix<size_t>>(resultMatrix, expectedResult, nullptr));
    
    /// cleanup
    DataObjectFactory::destroy(resultMatrix, expectedResult, pos_list);
}

TEST_CASE("BitmapPositionListConverter: Test case 2", TAG_KERNELS) {
    /// Data generation
    auto pos_list = genGivenVals<DenseMatrix<size_t>>(10, { 1, 1, 1, 1, 0, 0, 0, 1, 1, 1});
    
    DenseMatrix<size_t> * expectedResult;
    /// create expected result set
    {
        std::vector<size_t> er_col0_val;
        for (size_t outerLoop = 0; outerLoop < pos_list->getNumRows(); outerLoop++) {
            if(pos_list->get(outerLoop, 0) == 1) {
                    er_col0_val.push_back(outerLoop);
            }
        }
        size_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<size_t>>(size, er_col0_val);
    }
    
    /// test execution
    DenseMatrix<size_t> * resultMatrix = nullptr;

    bitmapPositionListConverter(resultMatrix, pos_list, nullptr);    
    
    /// test if result matches expected result
    CHECK(checkEq<DenseMatrix<size_t>>(resultMatrix, expectedResult, nullptr));
    resultMatrix->print(std::cout);
    expectedResult->print(std::cout);
    
    /// cleanup
    DataObjectFactory::destroy(resultMatrix, expectedResult, pos_list);
}