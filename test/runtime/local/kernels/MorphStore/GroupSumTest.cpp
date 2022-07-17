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

#include <runtime/local/kernels/MorphStore/groupsum.h>

TEST_CASE("Morphstore Groupsum: Test the operator with empty input", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    auto lhs_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    Frame * expectedResult;

    std::string groupLabels[] = {"R.a"};
    std::string resultLabels[] = {"R.a", "sum-R.a"};
    /// create expected result set
    {
        auto er_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        auto er_col1 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0, er_col1},
                resultLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    groupsum(resultFrame, f, groupLabels, 1, "R.a");

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);
}

TEST_CASE("Morphstore Groupsum: Test groupsum for one argument", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 33, 55, 60, 60, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    std::string groupLabels[] = {"R.a"};
    std::string resultLabels[] = {"R.a", "sum-R.a"};

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val = {0, 11, 20, 66, 55, 120, 88, 99};
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
            if(!(std::find(er_col0_val.begin(), er_col0_val.end(), lhs_col1->get(outerLoop, 0)) != er_col0_val.end())) {
                er_col0_val.push_back(lhs_col1->get(outerLoop, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0, er_col1},
                resultLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    groupsum(resultFrame, lhs, groupLabels, 1, "R.a");

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs);

}

TEST_CASE("Morphstore Groupsum: Test groupsum for more than one argument", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 11, 33, 44, 55, 55, 77, 77, 99});
    auto lhs_col2 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 11, 11, 55, 55, 60, 77, 77, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1, lhs_col2};
    std::string lhsLabels[] = {"R.idx", "R.a", "R.b"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    std::string groupLabels[] = {"R.a", "R.b"};
    std::string resultLabels[] = {"R.a","R.b", "sum-R.a"};

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val = {0, 11, 33, 44, 55, 55, 77, 99};
        std::vector<uint64_t> er_col1_val = {0, 11, 11, 55, 55, 60, 77, 99};
        std::vector<uint64_t> er_col2_val = {0, 22, 33, 44, 55, 55, 154, 99};
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        auto er_col2 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col2_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1, er_col2},
                resultLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1, er_col2);
        DataObjectFactory::destroy(lhs_col0, lhs_col1, lhs_col2);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    groupsum(resultFrame, lhs, groupLabels, 2, "R.a");

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs);

}

