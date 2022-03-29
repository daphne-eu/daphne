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

#include <tags.h>
#include <vector>
#include <cstdint>
#include <iostream>

#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <catch.hpp>

#include <runtime/local/kernels/ThetaJoin.h>


/// Test query Select * From R, S Where R.a = S.a
TEST_CASE("ThetaJoin: Test the equal (==) operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10,  {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) == rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a == S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::Equal};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}


/// Test query Select * From R, S Where R.a < S.a
TEST_CASE("ThetaJoin: Test the LessThan (<) operation", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) < rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a < S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::LessThan};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}

/// Test query Select * From R, S Where R.a <= S.a
TEST_CASE("ThetaJoin: Test the LessEqual (<=) operation", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) <= rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a <= S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::LessEqual};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}

/// Test query Select * From R, S Where R.a > S.a
TEST_CASE("ThetaJoin: Test the GreaterThan (>) operation", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) > rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a > S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::GreaterThan};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}

/// Test query Select * From R, S Where R.a >( S.a
TEST_CASE("ThetaJoin: Test the GreaterEqual (>=) operation", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) >= rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a >= S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::GreaterEqual};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}

/// Test query Select * From R, S Where R.a != S.a
TEST_CASE("ThetaJoin: Test the NonEqual (!=) operation", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10, {100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<int64_t> er_col3_val;
        std::vector<double> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(outerLoop, 0) != rhs_col1->get(innerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col2->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<int64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<double>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1, rhs_col2);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    /// R.a != S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    auto cmps = new CompareOperation[10]{CompareOperation::NotEqual};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}

/// Test query Select * From R, S Where R.idx = S.idx And R.a != S.a And R.b >= S.c
TEST_CASE("ThetaJoin: Test multiple conditions", TAG_KERNELS) {
    /// data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto lhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10,  {   0,   0,   0,   0,   0,   0,  40,   0,   0,  10});
    auto lhs_col2 = genGivenVals<DenseMatrix<uint64_t>>(10, {   1,   3,   5,   3,   1,   3,   5,   3,   1,   3});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1, lhs_col2};
    std::string lhsLabels[] = {"R.idx", "R.a", "R.b"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);
    
    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto rhs_col1 = genGivenVals<DenseMatrix<int64_t>>(10,  {100,  90,  80,  70,  60,  50,  40,  30,  20,  10});
    auto rhs_col2 = genGivenVals<DenseMatrix<double >>(10,  {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    auto rhs_col3 = genGivenVals<DenseMatrix<uint64_t>>(10, {2, 3, 2, 3, 2, 3, 2, 3, 2, 3});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2, rhs_col3};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b", "S.c"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<int64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<uint64_t> er_col3_val;
        std::vector<int64_t> er_col4_val;
        std::vector<double> er_col5_val;
        std::vector<uint64_t> er_col6_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < rhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (
                  /// R.idx == S.idx
                  lhs_col0->get(outerLoop, 0) == rhs_col0->get(innerLoop, 0) and
                  /// R.a != S.a
                  lhs_col1->get(outerLoop, 0) != rhs_col1->get(innerLoop, 0) and
                  /// R.b >= S.c
                  lhs_col2->get(outerLoop, 0) >= rhs_col3->get(innerLoop, 0)
                  ) {
                    er_col0_val.push_back(lhs_col0->get(outerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(outerLoop, 0));
                    er_col2_val.push_back(lhs_col2->get(outerLoop, 0));
                    er_col3_val.push_back(rhs_col0->get(innerLoop, 0));
                    er_col4_val.push_back(rhs_col1->get(innerLoop, 0));
                    er_col5_val.push_back(rhs_col2->get(innerLoop, 0));
                    er_col6_val.push_back(rhs_col3->get(innerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<int64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<int64_t>>(size, er_col4_val);
        auto er_col5 = genGivenVals<DenseMatrix<double>>(size, er_col5_val);
        auto er_col6 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col6_val);
        std::string labels[] = {"R.idx", "R.a", "R.b", "S.idx", "S.a", "S.b", "S.c"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{er_col0,er_col1,er_col2,er_col3,er_col4, er_col5, er_col6},
          labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4, er_col5, er_col6);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, lhs_col2, rhs_col0, rhs_col1, rhs_col2, rhs_col3);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 3;
    
    /// R.idx == S.idx && R.a != S.b && R.b >= S.c
    auto lhsQLabels = new const char*[10]{"R.idx", "R.a", "R.b"};
    auto rhsQLabels = new const char*[10]{"S.idx", "S.a", "S.c"};
    auto cmps = new CompareOperation[10]{CompareOperation::Equal, CompareOperation::NotEqual, CompareOperation::GreaterEqual};
    thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
    delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
    
    
    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}


/// Test query Select * From R, S Where R.X = S.Y with X, Y in [ui8, ui32, ui64, si8, si32, si64, f32, f64]
TEST_CASE("ThetaJoin: Test unequal value types", TAG_KERNELS) {
    /// data generation
    auto ui8  = genGivenVals<DenseMatrix<uint8_t >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto ui32 = genGivenVals<DenseMatrix<uint32_t>>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto ui64 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto si8  = genGivenVals<DenseMatrix<int8_t  >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto si32 = genGivenVals<DenseMatrix<int32_t >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto si64 = genGivenVals<DenseMatrix<int64_t >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto f32  = genGivenVals<DenseMatrix<float   >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    auto f64  = genGivenVals<DenseMatrix<double  >>(10, {   0,   1,   2,   3,   4,   5,   6,   7,   8,   9});
    std::vector<Structure *> cols = {ui8, ui32, ui64, si8, si32, si64, f32, f64};
    std::string lhsLabels[] = {"R.ui8","R.ui32","R.ui64","R.si8","R.si32","R.si64", "R.f32", "R.f64"};
    std::string rhsLabels[] = {"S.ui8","S.ui32","S.ui64","S.si8","S.si32","S.si64", "S.f32", "S.f64"};
    auto lhs = DataObjectFactory::create<Frame>(cols, lhsLabels);
    auto rhs = DataObjectFactory::create<Frame>(cols, rhsLabels);
    
    
    Frame * expectedResult;
    /// create expected result set
    {
        std::string labels[] = {"R.ui8","R.ui32","R.ui64","R.si8","R.si32","R.si64", "R.f32", "R.f64",
                                "S.ui8","S.ui32","S.ui64","S.si8","S.si32","S.si64", "S.f32", "S.f64"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
          std::vector<Structure*>{ui8, ui32, ui64, si8, si32, si64, f32, f64, ui8, ui32, ui64, si8, si32, si64, f32, f64},
          labels);
    }
    
    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;
    
    std::function<void(const std::string&, const std::string)> test = [&] (const std::string& lhsCol, const std::string& rhsCol){
        auto lhsQLabels = new const char*[10]{lhsCol.c_str()};
        auto rhsQLabels = new const char*[10]{rhsCol.c_str()};
        auto cmps = new CompareOperation[10]{CompareOperation::Equal};
        thetaJoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations, cmps, equations);
        delete[] lhsQLabels, delete[] rhsQLabels, delete[] cmps;
        
        /// test if result matches expected result
        CHECK(*resultFrame == *expectedResult);
        /// cleanup
        DataObjectFactory::destroy(resultFrame);
    };
    
    std::vector<std::string> types {"ui8", "ui32", "ui64", "si8", "si32", "si64", "f32", "f64"};
    
    /// check each possible combination
    for(auto& tlhs : types){
        for(auto& trhs : types){
            test("R." + tlhs, "S." + trhs);
        }
    }
    
    
    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);
}
