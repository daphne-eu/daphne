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

#include <runtime/local/kernels/MorphStore/sum.h>

TEST_CASE("Morphstore Sum: Test the AggSum operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto f = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        uint64_t sum = 0;
        std::string resultLabels[] = {"Agg_sum"};
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            sum += lhs_col1->get(i, 0);
        }
        er_col0_val.push_back(sum);
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0},
                resultLabels);
        /// cleanup
        DataObjectFactory::destroy(er_col0);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    agg_sum(resultFrame, f, "R.a");

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, f);

}