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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ExtractCol.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>


/**
 * @brief Runs the extractCol-kernel with small input data and performs various
 * checks.
 */
TEMPLATE_TEST_CASE("ExtractCol - Frame", TAG_KERNELS, int64_t, size_t) { // NOLINT(cert-err58-cpp)
    using VTSel = TestType;
    using DTSel = DenseMatrix<VTSel>;
    
    using DT0 = DenseMatrix<double>;
    using DT1 = DenseMatrix<int32_t>;
    using DT2 = DenseMatrix<uint64_t>;
    
    const size_t numRows = 5;
    
    auto c0 = genGivenVals<DT0>(numRows, {0.0, 1.1, 2.2, 3.3, 4.4});
    auto c1 = genGivenVals<DT1>(numRows, {0, -10, -20, -30, -40});
    auto c2 = genGivenVals<DT2>(numRows, {0, 1, 2, 3, 4});

    std::vector<Structure *> colMats = {c0, c1, c2};
    std::string labels[] = {"aaa", "bbb", "ccc"};
    auto arg = DataObjectFactory::create<Frame>(colMats, labels);
    size_t numColExp;
    
    Frame* res{};
    Frame* exp{};
    DTSel* sel{};

    SECTION("selecting nothing") {
        sel = DataObjectFactory::create<DTSel>(0, 1, false);
        exp = DataObjectFactory::create<Frame>(arg, 0, arg->getNumRows(), 0, nullptr);
    }
    SECTION("selecting some, once, in-order") {
        numColExp = 2;
        sel = genGivenVals<DTSel>(numColExp, {0, 2});
        std::vector<Structure *> colMatsExp = {c0, c2};
        std::string labelsExp[] = {"aaa", "ccc"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
    }
    SECTION("selecting everything, once, in-order") {
        numColExp = 3;
        sel = genGivenVals<DTSel>(numColExp, {0, 1, 2});
        std::vector<Structure *> colMatsExp = {c0, c1, c2};
        std::string labelsExp[] = {"aaa", "bbb", "ccc"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
    }
    SECTION("selecting everything, once, permuted") {
        numColExp = 3;
        sel = genGivenVals<DTSel>(numColExp, {2, 0, 1});
        std::vector<Structure *> colMatsExp = {c2, c0, c1};
        std::string labelsExp[] = {"ccc", "aaa", "bbb"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
    }
    SECTION("selecting some, repeated") {
        numColExp = 8;
        sel = genGivenVals<DTSel>(numColExp, {1, 2, 2, 0, 1, 0, 1, 2});
        std::vector<Structure *> colMatsExp = {c1, c2, c2, c0, c1, c0, c1, c2};
        std::string labelsExp[] = {"bbb", "ccc", "col_2", "aaa", "col_4", "col_5", "col_6", "col_7"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
    }
    
    extractCol<Frame, Frame, DTSel>(res, arg, sel, nullptr);

    CHECK(*res == *exp);
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}
