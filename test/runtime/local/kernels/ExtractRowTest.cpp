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
#include <runtime/local/kernels/ExtractRow.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <type_traits>
#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix

/**
 * @brief Runs the extractRow-kernel with small input data and performs various
 * checks.
 */
TEMPLATE_TEST_CASE("ExtractRow - Frame", TAG_KERNELS, double, int64_t, uint32_t) { // NOLINT(cert-err58-cpp)
    using VTSel = TestType;
    using DTSel = DenseMatrix<VTSel>;
    
    using DT0 = DenseMatrix<double>;
    using DT1 = DenseMatrix<int32_t>;
    using DT2 = DenseMatrix<uint64_t>;
    
    const size_t numCols = 3;
    const size_t numRows = 5;
    
    auto c0 = genGivenVals<DT0>(numRows, {0.0, 1.1, 2.2, 3.3, 4.4});
    auto c1 = genGivenVals<DT1>(numRows, {0, -10, -20, -30, -40});
    auto c2 = genGivenVals<DT2>(numRows, {0, 1, 2, 3, 4});
    std::vector<Structure *> colMats = {c0, c1, c2};
    std::string labels[] = {"aaa", "bbb", "ccc"};
    auto arg = DataObjectFactory::create<Frame>(colMats, labels);
    
    Frame* res{};
    DTSel* sel{};
    size_t numRowsExp{};
    DenseMatrix<double>* c0Exp{};
    DenseMatrix<int32_t>* c1Exp{};
    DenseMatrix<uint64_t>* c2Exp{};
    SECTION("selecting nothing") {
        numRowsExp = 0;
        sel = DataObjectFactory::create<DTSel>(0, 1, false);
        c0Exp = DataObjectFactory::create<DT0>(0, 1, false);
        c1Exp = DataObjectFactory::create<DT1>(0, 1, false);
        c2Exp = DataObjectFactory::create<DT2>(0, 1, false);
    }
    SECTION("selecting some, once, in-order") {
        numRowsExp = 3;
        sel = genGivenVals<DTSel>(numRowsExp, {1, 3, 4});
        c0Exp = genGivenVals<DT0>(numRowsExp, {1.1, 3.3, 4.4});
        c1Exp = genGivenVals<DT1>(numRowsExp, {-10, -30, -40});
        c2Exp = genGivenVals<DT2>(numRowsExp, {1, 3, 4});
    }
    SECTION("selecting everything, once, in-order") {
        numRowsExp = 5;
        sel = genGivenVals<DTSel>(numRows, {0, 1, 2, 3, 4});
        c0Exp = c0;
        c1Exp = c1;
        c2Exp = c2;
    }
    SECTION("selecting everything, once, permuted") {
        numRowsExp = 5;
        sel = genGivenVals<DTSel>(numRowsExp, {3, 1, 4, 0, 2});
        c0Exp = genGivenVals<DT0>(numRowsExp, {3.3, 1.1, 4.4, 0.0, 2.2});
        c1Exp = genGivenVals<DT1>(numRowsExp, {-30, -10, -40, 0, -20});
        c2Exp = genGivenVals<DT2>(numRowsExp, {3, 1, 4, 0, 2});
    }
    SECTION("selecting some, repeated") {
        numRowsExp = 8;
        sel = genGivenVals<DTSel>(numRowsExp, {3, 2, 2, 0, 1, 4, 4, 2});
        c0Exp = genGivenVals<DT0>(numRowsExp, {3.3, 2.2, 2.2, 0.0, 1.1, 4.4, 4.4, 2.2});
        c1Exp = genGivenVals<DT1>(numRowsExp, {-30, -20, -20, 0, -10, -40, -40, -20});
        c2Exp = genGivenVals<DT2>(numRowsExp, {3, 2, 2, 0, 1, 4, 4, 2});
    }
    
    extractRow<Frame, Frame, VTSel>(res, arg, sel, nullptr);
    
    // Check expected #rows.
    REQUIRE(res->getNumRows() == numRowsExp);
    // Check that #columns, schema, and labels remain unchanged.
    REQUIRE(res->getNumCols() == numCols);
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
TEMPLATE_PRODUCT_TEST_CASE("ExtractRow - Matrix double naive", TAG_KERNELS, (DATA_TYPES), (uint32_t)) { // NOLINT(cert-err58-cpp)
    using DT = typename TestType::template WithValueType<double>;
    using VT = typename TestType::VT;
    using DTSel = DenseMatrix<VT>;

    auto argMatrix = genGivenVals<DT>(4, {
        1.0,  10.0, 3.0, 7.0, 7.0, 7.0,
        17.0, 1.0,  2.0, 3.0, 7.0, 7.0,
        7.0,  7.0,  1.0, 2.0, 3.0, 7.0,
        7.0,  7.0,  7.0, 1.0, 2.0, 3.0
    });
    auto selMatrix = genGivenVals<DTSel>(4, {
        0,
        1,
        2,
        3
    });
    DT * resMatrix = nullptr;

    extractRow<DT, DT, VT>(resMatrix, argMatrix, selMatrix, nullptr);
    CHECK(*resMatrix == *argMatrix);
    DataObjectFactory::destroy(argMatrix, selMatrix, resMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("ExtractRow - Matrix double expanding", TAG_KERNELS, (DATA_TYPES), (uint32_t)) { // NOLINT(cert-err58-cpp)
    using DT = typename TestType::template WithValueType<double>;
    using VT = typename TestType::VT;
    using DTSel = DenseMatrix<VT>;

    auto argMatrix = genGivenVals<DT>(4, {
        1.0,  10.0, 3.0, 7.0, 7.0, 7.0,
        17.0, 1.0,  2.0, 3.0, 7.0, 7.0,
        7.0,  7.0,  1.0, 2.0, 3.0, 7.0,
        7.0,  7.0,  7.0, 1.0, 2.0, 3.0
    });
    auto selMatrix = genGivenVals<DTSel>(8, {
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3
    });
    auto expMatrix = genGivenVals<DT>(8, {
        1.0,  10.0, 3.0,  7.0,  7.0,  7.0,
        17.0, 1.0,  2.0,  3.0,  7.0,  7.0,
        7.0,  7.0,  1.0,  2.0,  3.0,  7.0,
        7.0,  7.0,  7.0,  1.0,  2.0,  3.0,
        1.0,  10.0, 3.0,  7.0,  7.0,  7.0,
        17.0, 1.0,  2.0,  3.0,  7.0,  7.0,
        7.0,  7.0,  1.0,  2.0,  3.0,  7.0,
        7.0,  7.0,  7.0,  1.0,  2.0,  3.0
    });
    DT * resMatrix = nullptr;

    extractRow<DT, DT, VT>(resMatrix, argMatrix, selMatrix, nullptr);
    CHECK(*resMatrix == *expMatrix);
    DataObjectFactory::destroy(argMatrix, selMatrix, expMatrix, resMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("ExtractRow - Matrix int unordered", TAG_KERNELS, (DATA_TYPES), (uint32_t)) { // NOLINT(cert-err58-cpp)
    using DT = typename TestType::template WithValueType<int64_t>;
    using VT = typename TestType::VT;
    using DTSel = DenseMatrix<VT>;

    auto argMatrix = genGivenVals<DT>(4, {
        1,  10, 3, 7, 7, 7,
        17, 1,  2, 3, 7, 7,
        7,  7,  1, 2, 3, 7,
        7,  7,  7, 1, 2, 3
    }); 
    auto selMatrix = genGivenVals<DTSel>(4, {
        2,
        3,
        0,
        1
    });
    auto expMatrix =  genGivenVals<DT>(4, {
        7,  7,  1, 2, 3, 7,
        7,  7,  7, 1, 2, 3,
        1,  10, 3, 7, 7, 7,
        17, 1,  2, 3, 7, 7
    });
    DT * resMatrix = nullptr;

    extractRow<DT, DT, VT>(resMatrix, argMatrix, selMatrix, nullptr);
    CHECK(*resMatrix == *expMatrix);
    DataObjectFactory::destroy(argMatrix, selMatrix, expMatrix, resMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("ExtractRow - Matrix int repeated with initialized resMatrix", TAG_KERNELS, (DATA_TYPES), (uint32_t)) { // NOLINT(cert-err58-cpp)
    using DT = typename TestType::template WithValueType<int64_t>;
    using VT = typename TestType::VT;
    using DTSel = DenseMatrix<VT>;

    auto argMatrix = genGivenVals<DT>(4, {
        1,  10, 3, 7, 7, 7,
        17, 1,  2, 3, 7, 7,
        7,  7,  1, 2, 3, 7,
        7,  7,  7, 1, 2, 3
    });
    auto selMatrix = genGivenVals<DTSel>(4, {
        1,
        1,
        1, 
        1
    });
    auto* expMatrix = genGivenVals<DT>(4, {
        17, 1, 2, 3, 7, 7,
        17, 1, 2, 3, 7, 7,
        17, 1, 2, 3, 7, 7,
        17, 1, 2, 3, 7, 7
    });
    auto* resMatrix = genGivenVals<DT>(4, {
        7,  0,  0,  2,  3,  7,
        7,  7,  7,  1,  2,  3,
        1,  10, 3,  -7, -7, 7,
        17, 1,  2,  3,  7,  7
    });

    extractRow<DT, DT, VT>(resMatrix, argMatrix, selMatrix, nullptr);
    CHECK(*resMatrix == *expMatrix);
    DataObjectFactory::destroy(argMatrix, selMatrix, expMatrix, resMatrix);
}

TEMPLATE_PRODUCT_TEST_CASE("ExtractRow - Matrix boundary checking", TAG_KERNELS, (DATA_TYPES), (int32_t, double)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTSel = DenseMatrix<VT>;

    auto argMatrix = genGivenVals<DT>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    DTSel * selMatrix = nullptr;
    DT * resMatrix = nullptr;

    SECTION("sel out of bounds - negative") {
        selMatrix = genGivenVals<DTSel>(3, {
            -1,
            2,
            2
        });
    }
    SECTION("sel out of bounds - too high") {
        selMatrix = genGivenVals<DTSel>(3, {
            0,
            2,
            3
        });
    }

    REQUIRE_THROWS_AS((extractRow<DT, DT, VT>(resMatrix, argMatrix, selMatrix, nullptr)), std::out_of_range);
    DataObjectFactory::destroy(argMatrix, selMatrix, resMatrix);
}

TEMPLATE_TEST_CASE("ExtractRow - Frame boundary checking", TAG_KERNELS, int32_t, double) {
    using VTSel = TestType;
    using DTSel = DenseMatrix<VTSel>;
    
    using DT0 = DenseMatrix<double>;
    using DT1 = DenseMatrix<int32_t>;
    using DT2 = DenseMatrix<uint64_t>;
    
    auto c0 = genGivenVals<DT0>(5, {0.0, 1.1, 2.2, 3.3, 4.4});
    auto c1 = genGivenVals<DT1>(5, {0, -10, -20, -30, -40});
    auto c2 = genGivenVals<DT2>(5, {0, 1, 2, 3, 4});
    std::vector<Structure *> colMats = {c0, c1, c2};
    std::string labels[] = {"aaa", "bbb", "ccc"};
    auto arg = DataObjectFactory::create<Frame>(colMats, labels);

    DTSel * selMatrix = nullptr;
    Frame * res = nullptr;

    SECTION("sel out of bounds - negative") {
        selMatrix = genGivenVals<DTSel>(3, {
            -1,
            2,
            2,
        });
    }
    SECTION("sel out of bounds - too high") {
        selMatrix = genGivenVals<DTSel>(3, {
            0,
            2,
            5,
        });
    }

    REQUIRE_THROWS_AS((extractRow<Frame, Frame, VTSel>(res, arg, selMatrix, nullptr)), std::out_of_range);
    DataObjectFactory::destroy(arg, selMatrix, res);
}