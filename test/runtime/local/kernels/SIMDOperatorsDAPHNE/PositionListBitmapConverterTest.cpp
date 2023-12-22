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
#include <runtime/local/kernels/SIMDOperatorsDAPHNE/PositionListBitmapConverter.h>

#include <mlir/IR/Location.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

TEST_CASE("PositionListBitmapConverter: Test case 1", TAG_KERNELS) {
    /// Data generation
    auto pos_list = genGivenVals<DenseMatrix<size_t>>(5, { 0,  3,   7,  8,  9});
    size_t origLength = 10;
    
    DenseMatrix<size_t> * expectedResult;
    /// create expected result set
    {
        std::vector<size_t> er_col0_val;
        size_t current_position = 0;
        for (size_t outerLoop = 0; outerLoop < origLength; outerLoop++) {
            if(current_position == pos_list->getNumRows()) {
                    er_col0_val.push_back(0);
            } else if (outerLoop == pos_list->get(current_position, 0)) {
                er_col0_val.push_back(1);
                current_position++;
            } else {
                er_col0_val.push_back(0);
            }
        }
        size_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<size_t>>(size, er_col0_val);
    }
    
    /// test execution
    DenseMatrix<size_t> * resultMatrix = nullptr;

    positionListBitmapConverter(resultMatrix, pos_list, origLength, nullptr);    
    
    /// test if result matches expected result
    CHECK(checkEq<DenseMatrix<size_t>>(resultMatrix, expectedResult, nullptr));
    
    /// cleanup
    DataObjectFactory::destroy(resultMatrix, expectedResult, pos_list);
}

TEST_CASE("PositionListBitmapConverter: Test case 2", TAG_KERNELS) {
    /// Data generation
    auto pos_list = genGivenVals<DenseMatrix<size_t>>(5, { 2,  3,   4,  5,  6});
    size_t origLength = 10;  
    
    DenseMatrix<size_t> * expectedResult;
    /// create expected result set
    {
        std::vector<size_t> er_col0_val;
        size_t current_position = 0;
        for (size_t outerLoop = 0; outerLoop < origLength; outerLoop++) {
            if(current_position == pos_list->getNumRows()) {
                    er_col0_val.push_back(0);
            } else if (outerLoop == pos_list->get(current_position, 0)) {
                er_col0_val.push_back(1);
                current_position++;
            } else {
                er_col0_val.push_back(0);
            }
        }
        size_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<size_t>>(size, er_col0_val);
    }
    
    /// test execution
    DenseMatrix<size_t> * resultMatrix = nullptr;

    positionListBitmapConverter(resultMatrix, pos_list, origLength, nullptr);    
    
    /// test if result matches expected result
    CHECK(checkEq<DenseMatrix<size_t>>(resultMatrix, expectedResult, nullptr));
    
    /// cleanup
    DataObjectFactory::destroy(resultMatrix, expectedResult, pos_list);
}

TEST_CASE("PositionListBitmapConverter: Test case empty position list", TAG_KERNELS) {
    /// Data generation
    DenseMatrix<size_t> * pos_list = DataObjectFactory::create<DenseMatrix<size_t>>(0, 0, false);
    size_t origLength = 10;  
    
    DenseMatrix<size_t> * expectedResult;
    /// create expected result set
    {
        std::vector<size_t> er_col0_val;
        size_t current_position = 0;
        for (size_t outerLoop = 0; outerLoop < origLength; outerLoop++) {
            if(current_position == pos_list->getNumRows()) {
                    er_col0_val.push_back(0);
            } else if (outerLoop == pos_list->get(current_position, 0)) {
                er_col0_val.push_back(1);
                current_position++;
            } else {
                er_col0_val.push_back(0);
            }
        }
        size_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<size_t>>(size, er_col0_val);
    }
    
    /// test execution
    DenseMatrix<size_t> * resultMatrix = nullptr;

    positionListBitmapConverter(resultMatrix, pos_list, origLength, nullptr);    
    
    /// test if result matches expected result
    CHECK(checkEq<DenseMatrix<size_t>>(resultMatrix, expectedResult, nullptr));
    
    /// cleanup
    DataObjectFactory::destroy(resultMatrix, expectedResult, pos_list);
}