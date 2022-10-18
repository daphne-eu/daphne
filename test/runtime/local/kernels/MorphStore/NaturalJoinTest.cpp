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

#include <runtime/local/kernels/MorphStore/naturaljoin.h>

TEST_CASE("Morphstore Naturaljoin: Test the operator with empty input", TAG_KERNELS) {
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
        auto er_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col2 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col3 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1,er_col2, er_col3},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;

    /// R.a == S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};

    naturaljoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations);
    delete[] lhsQLabels, delete[] rhsQLabels;

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Naturaljoin: Test the operator with empty output", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 22, 33, 44, 55, 66, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 20, 40, 30, 60, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        auto er_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col2 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col3 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1,er_col2, er_col3},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;

    /// R.a == S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};

    naturaljoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations);
    delete[] lhsQLabels, delete[] rhsQLabels;

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Naturaljoin: Test the Equals operation with one join condition where R.a = S.a", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 10, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 20, 40, 30, 60, 10});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1};
    std::string rhsLabels[] = {"S.idx", "S.a"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<uint64_t> er_col3_val;
        for (uint64_t outerLoop = 0; outerLoop < rhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < lhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(innerLoop, 0) == rhs_col1->get(outerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(innerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(innerLoop, 0));
                    er_col2_val.push_back(rhs_col0->get(outerLoop, 0));
                    er_col3_val.push_back(rhs_col1->get(outerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col3_val);
        std::string labels[] = {"R.idx", "R.a", "S.idx", "S.a"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1,er_col2, er_col3},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 1;

    /// R.a == S.a
    auto lhsQLabels = new const char*[10]{"R.a"};
    auto rhsQLabels = new const char*[10]{"S.a"};
    naturaljoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations);
    delete[] lhsQLabels, delete[] rhsQLabels;

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Naturaljoin: Test the Equals operation with two join conditions, here R.a=S.a AND R.b = S.a", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    auto lhs_col2 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 22, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1, lhs_col2};
    std::string lhsLabels[] = {"R.idx", "R.a", "R.b"};
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
        std::vector<uint64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<uint64_t> er_col3_val;
        std::vector<uint64_t> er_col4_val;
        for (uint64_t outerLoop = 0; outerLoop < rhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < lhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(innerLoop, 0) == rhs_col1->get(outerLoop, 0) and lhs_col2->get(innerLoop, 0) == rhs_col1->get(outerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(innerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(innerLoop, 0));
                    er_col2_val.push_back(lhs_col2->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col0->get(outerLoop, 0));
                    er_col4_val.push_back(rhs_col1->get(outerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col4_val);
        std::string labels[] = {"R.idx", "R.a", "R.b", "S.idx", "S.a"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1,er_col2, er_col3, er_col4},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, lhs_col2, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 2;

    /// R.a == S.a and R.b == S.a
    auto lhsQLabels = new const char*[10]{"R.a", "R.b"};
    auto rhsQLabels = new const char*[10]{"S.a", "S.a"};

    naturaljoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations);
    delete[] lhsQLabels, delete[] rhsQLabels;

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}

TEST_CASE("Morphstore Naturaljoin: Test the Equals operation with more than two join conditions, here R.a == S.a AND R.b == S.a AND R.b == S.b", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 30, 44, 55, 60, 77, 88, 99});
    auto lhs_col2 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1, lhs_col2};
    std::string lhsLabels[] = {"R.idx", "R.a", "R.b"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, {   0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto rhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});
    auto rhs_col2 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 22, 33, 60, 61, 66, 77, 88, 99});
    std::vector<Structure *> rhsCols = {rhs_col0, rhs_col1, rhs_col2};
    std::string rhsLabels[] = {"S.idx", "S.a", "S.b"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;
        std::vector<uint64_t> er_col2_val;
        std::vector<uint64_t> er_col3_val;
        std::vector<uint64_t> er_col4_val;
        std::vector<uint64_t> er_col5_val;
        for (uint64_t outerLoop = 0; outerLoop < rhs_col0->getNumRows(); ++ outerLoop) {
            for (uint64_t innerLoop = 0; innerLoop < lhs_col0->getNumRows(); ++ innerLoop) {
                /// condition to check
                if (lhs_col1->get(innerLoop, 0) == rhs_col1->get(outerLoop, 0) and lhs_col2->get(innerLoop, 0) == rhs_col1->get(outerLoop, 0)
                    and lhs_col2->get(innerLoop, 0) == rhs_col2->get(outerLoop, 0)) {
                    er_col0_val.push_back(lhs_col0->get(innerLoop, 0));
                    er_col1_val.push_back(lhs_col1->get(innerLoop, 0));
                    er_col2_val.push_back(lhs_col2->get(innerLoop, 0));
                    er_col3_val.push_back(rhs_col0->get(outerLoop, 0));
                    er_col4_val.push_back(rhs_col1->get(outerLoop, 0));
                    er_col5_val.push_back(rhs_col2->get(outerLoop, 0));
                }
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        auto er_col3 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col3_val);
        auto er_col4 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col4_val);
        auto er_col5 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col5_val);
        std::string labels[] = {"R.idx", "R.a", "R.b", "S.idx", "S.a", "S.b"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1,er_col2, er_col3, er_col4, er_col5},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2, er_col3, er_col4, er_col5);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, lhs_col2, rhs_col0, rhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;
    uint64_t equations = 3;

    /// R.a == S.a and R.b == S.a and R.b == S.b
    auto lhsQLabels = new const char*[10]{"R.a", "R.b", "R.b"};
    auto rhsQLabels = new const char*[10]{"S.a", "S.a", "S.b"};

    naturaljoin(resultFrame, lhs, rhs, lhsQLabels, equations, rhsQLabels, equations);
    delete[] lhsQLabels, delete[] rhsQLabels;

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, rhs);

}
