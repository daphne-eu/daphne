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
#include <runtime/local/kernels/FilterRow.h>
#include <runtime/local/kernels/RandMatrix.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("FilterRow", TAG_KERNELS, (DenseMatrix, Matrix), (double, int64_t, uint32_t)) {
    using DT = TestType;
    using VTArg = typename DT::VT;
    using VTSel = int64_t;
    using DTSel = DenseMatrix<VTSel>;
    using DTEmpty = typename std::conditional<
                        std::is_same<DT, Matrix<VTArg>>::value,
                        DenseMatrix<VTArg>,
                        DT
                    >::type;

    auto arg = genGivenVals<DT>(5, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    });

    DTSel * sel = nullptr;
    DT * exp = nullptr;
    SECTION("bit vector empty") {
        sel = genGivenVals<DTSel>(5, {0, 0, 0, 0, 0});
        exp = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 3, false));
    }
    SECTION("bit vector contiguous 0") {
        sel = genGivenVals<DTSel>(5, {0, 0, 1, 1, 1});
        exp = genGivenVals<DT>(3, {
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
        });
    }
    SECTION("bit vector contiguous 1") {
        sel = genGivenVals<DTSel>(5, {1, 1, 1, 0, 0});
        exp = genGivenVals<DT>(3, {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        });
    }
    SECTION("bit vector mixed") {
        sel = genGivenVals<DTSel>(5, {0, 1, 1, 0, 1});
        exp = genGivenVals<DT>(3, {
            4, 5, 6,
            7, 8, 9,
            13, 14, 15,
        });
    }
    SECTION("bit vector full") {
        sel = genGivenVals<DTSel>(5, {1, 1, 1, 1, 1});
        exp = genGivenVals<DT>(5, {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
        });
    }

    DT * res = nullptr;
    filterRow<DT, DT, VTSel>(res, arg, sel, nullptr);

    CHECK(*res == *exp);

    DataObjectFactory::destroy(arg, sel, exp, res);
}

/**
 * @brief Runs the filterRow-kernel with small input data and performs various
 * checks.
 */
TEMPLATE_TEST_CASE("FilterRow - Frame", TAG_KERNELS, double, int64_t, uint32_t) { // NOLINT(cert-err58-cpp)
    using VTSel = TestType;
    using DTSel = DenseMatrix<VTSel>;
    
    using DT0 = DenseMatrix<double>;
    using DT1 = DenseMatrix<int32_t>;
    using DT2 = DenseMatrix<uint64_t>;
    
    const size_t numCols = 3;
    const size_t numRows = 5;
    
    auto c0 = genGivenVals<DT0>(numRows, {1.1, 2.2, 3.3, 4.4, 5.5});
    auto c1 = genGivenVals<DT1>(numRows, {-10, -20, -30, -40, -50});
    auto c2 = genGivenVals<DT2>(numRows, {1, 2, 3, 4, 5});
    std::vector<Structure *> colMats = {c0, c1, c2};
    std::string labels[] = {"aaa", "bbb", "ccc"};
    auto arg = DataObjectFactory::create<Frame>(colMats, labels);
    
    Frame* res{};
    DTSel* sel{};
    size_t numRowsExp{};
    DenseMatrix<double> * c0Exp{};
    DenseMatrix<int32_t> * c1Exp{};
    DenseMatrix<uint64_t> * c2Exp{};
    SECTION("selecting nothing") {
        sel = genGivenVals<DTSel>(numRows, {0, 0, 0, 0, 0});
        numRowsExp = 0;
        c0Exp = DataObjectFactory::create<DT0>(0, 1, false);
        c1Exp = DataObjectFactory::create<DT1>(0, 1, false);
        c2Exp = DataObjectFactory::create<DT2>(0, 1, false);
    }
    SECTION("selecting some, but not all") {
        sel = genGivenVals<DTSel>(numRows, {0, 1, 0, 1, 1});
        numRowsExp = 3;
        c0Exp = genGivenVals<DT0>(numRowsExp, {2.2, 4.4, 5.5});
        c1Exp = genGivenVals<DT1>(numRowsExp, {-20, -40, -50});
        c2Exp = genGivenVals<DT2>(numRowsExp, {2, 4, 5});
    }
    SECTION("selecting everything") {
        sel = genGivenVals<DTSel>(numRows, {1, 1, 1, 1, 1});
        numRowsExp = 5;
        c0Exp = c0;
        c1Exp = c1;
        c2Exp = c2;
    }
    
    filterRow<Frame, Frame, VTSel>(res, arg, sel, nullptr);
    
    // Check expected #rows.
    CHECK(res->getNumRows() == numRowsExp);
    // Check that #columns, schema, and labels remain unchanged.
    CHECK(res->getNumCols() == numCols);
    for(size_t i = 0; i < numCols; i++) {
        CHECK(res->getSchema()[i] == arg->getSchema()[i]);
        CHECK(res->getLabels()[i] == labels[i]);
    }
    // Check the data.
    CHECK(*(res->getColumn<double>(0)) == *c0Exp);
    CHECK(*(res->getColumn<int32_t>(1)) == *c1Exp);
    CHECK(*(res->getColumn<uint64_t>(2)) == *c2Exp);
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    if(c0 != c0Exp) {
        DataObjectFactory::destroy(c0Exp);
        DataObjectFactory::destroy(c1Exp);
        DataObjectFactory::destroy(c2Exp);
    }
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(res);
}

/**
 * @brief Runs the filterRow-kernel with large random input data only to check
 * if it returns the expected number of rows and doesn't crash.
 */
TEMPLATE_TEST_CASE("FilterRow (large input) - Frame", TAG_KERNELS, double, int64_t, uint32_t) { // NOLINT(cert-err58-cpp)
    using VTSel = TestType;
    using DTSel = DenseMatrix<VTSel>;
    
    using DT0 = DenseMatrix<double>;
    using DT1 = DenseMatrix<int32_t>;
    using DT2 = DenseMatrix<uint64_t>;
    
    const size_t numRows = 10000;
    
    auto c0 = DataObjectFactory::create<DT0>(numRows, 1, false);
    auto c1 = DataObjectFactory::create<DT1>(numRows, 1, false);
    auto c2 = DataObjectFactory::create<DT2>(numRows, 1, false);
    std::vector<Structure *> colMats = {c0, c1, c2};
    auto arg = DataObjectFactory::create<Frame>(colMats, nullptr);
    
    // Randomly generate the selection with a share of 1s equal to selectivity.
    const double selectivity = 0.01;
    DTSel * sel = nullptr;
    randMatrix<DTSel, VTSel>(
            sel, numRows, 1, VTSel(1), VTSel(1), selectivity, -1, nullptr
    );
    
    Frame * res = nullptr;
    filterRow<Frame, Frame, VTSel>(res, arg, sel, nullptr);
    
    // Check expected #rows.
    const auto numRowsExp = static_cast<size_t>(round(selectivity * numRows));
    CHECK(res->getNumRows() == numRowsExp);
    
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(arg);
    DataObjectFactory::destroy(res);
}