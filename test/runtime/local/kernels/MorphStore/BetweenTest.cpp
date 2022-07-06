/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/
#include <mlir/IR/Location.h>

#include <tags.h>
#include <vector>
#include <cstdint>
#include <iostream>

#include <catch.hpp>

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>


#include <runtime/local/kernels/MorphStore/select.h>
#include <runtime/local/kernels/MorphStore/between.h>

TEST_CASE("Morphstore Between: Test the operator with empty input", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    auto lhs_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 33;
    size_t upperBound = 70;

    Frame * expectedResult;
    /// create expected result set
    {
        auto er_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterThan, upperBound, CompareOperation::LessThan);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}

TEST_CASE("Morphstore Between: Test the operator with empty output", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 100;
    size_t upperBound = 150;

    Frame * expectedResult;
    /// create expected result set
    {
        auto er_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterThan, upperBound, CompareOperation::LessThan);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}

TEST_CASE("Morphstore Between: Test the Greater (>) and Less (<) operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 33;
    size_t upperBound = 70;

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            /// condition to check
            if (lhs_col1->get(i, 0) > lowerBound and lhs_col1->get(i, 0) < upperBound) {
                er_col0_val.push_back(lhs_col0->get(i, 0));
                er_col1_val.push_back(lhs_col1->get(i, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterThan, upperBound, CompareOperation::LessThan);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}

TEST_CASE("Morphstore Between: Test the GreaterEqual (>=) and Less (<) operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 33;
    size_t upperBound = 70;

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            /// condition to check
            if (lhs_col1->get(i, 0) >= lowerBound and lhs_col1->get(i, 0) < upperBound) {
                er_col0_val.push_back(lhs_col0->get(i, 0));
                er_col1_val.push_back(lhs_col1->get(i, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterEqual, upperBound, CompareOperation::LessThan);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}

TEST_CASE("Morphstore Between: Test the Greater (>) and LessEqual (<) operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 33;
    size_t upperBound = 77;

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            /// condition to check
            if (lhs_col1->get(i, 0) > lowerBound and lhs_col1->get(i, 0) <= upperBound) {
                er_col0_val.push_back(lhs_col0->get(i, 0));
                er_col1_val.push_back(lhs_col1->get(i, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterThan, upperBound, CompareOperation::LessEqual);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}

TEST_CASE("Morphstore Between: Test the GreaterEqual (>=) and LessEqual (<=) operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    size_t lowerBound = 33;
    size_t upperBound = 70;

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            /// condition to check
            if (lhs_col1->get(i, 0) >= lowerBound and lhs_col1->get(i, 0) <= upperBound) {
                er_col0_val.push_back(lhs_col0->get(i, 0));
                er_col1_val.push_back(lhs_col1->get(i, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                lhsLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    between(resultFrame, f, "R.a", lowerBound, CompareOperation::GreaterEqual, upperBound, CompareOperation::LessEqual);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}


