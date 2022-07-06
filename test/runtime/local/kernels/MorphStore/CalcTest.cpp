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

#include <runtime/local/kernels/MorphStore/calc.h>

TEST_CASE("Morphstore Calc: Test the operator with empty input", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    auto lhs_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    auto rhs_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        auto er_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        std::string labels[] = {"Calc"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    calc(resultFrame, lhs, rhs, "R.a", "S.a", CalcOperation::Add);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Add Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
                er_col0_val.push_back(lhs_col1->get(outerLoop, 0) + rhs_col1->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        std::string labels[] = {"Calc"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    calc(resultFrame, lhs, rhs, "R.a", "S.a", CalcOperation::Add);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Sub Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            er_col0_val.push_back(lhs_col1->get(outerLoop, 0) - rhs_col1->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        std::string labels[] = {"Calc"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    calc(resultFrame, lhs, rhs, "R.a", "S.a", CalcOperation::Sub);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Mul Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            er_col0_val.push_back(lhs_col1->get(outerLoop, 0) * rhs_col1->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        std::string labels[] = {"Calc"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    calc(resultFrame, lhs, rhs, "R.a", "S.a", CalcOperation::Mul);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Div Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 1, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            er_col0_val.push_back(lhs_col1->get(outerLoop, 0) / rhs_col1->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        std::string labels[] = {"Calc"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    calc(resultFrame, lhs, rhs, "R.a", "S.a", CalcOperation::Div);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}
