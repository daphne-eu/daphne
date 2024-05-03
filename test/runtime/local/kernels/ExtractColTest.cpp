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
#include <type_traits>
#include <vector>

#include <cstdint>

#define TEST_NAME(name) "ExtractCol - (" name ")"
#define DATA_TYPES DenseMatrix, Matrix

template<typename DT, typename DTSel>
void checkExtractCol(DT * res, DT * arg, DTSel * sel, DT * exp) {
    extractCol<DT, DT, DTSel>(res, arg, sel, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res, exp);
}

template<typename DT, typename DTSel>
void checkExtractColThrow(DT * res, DT * arg, DTSel * sel) {
    REQUIRE_THROWS_AS((extractCol<DT, DT, DTSel>(res, arg, sel, nullptr)), std::out_of_range);
}

/**
 * @brief Runs the extractCol-kernel with small input data and performs various
 * checks.
 */
TEMPLATE_TEST_CASE(TEST_NAME("Frame"), TAG_KERNELS, int64_t, size_t) { // NOLINT(cert-err58-cpp)
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
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting some, once, in-order") {
        numColExp = 2;
        sel = genGivenVals<DTSel>(numColExp, {0, 2});
        std::vector<Structure *> colMatsExp = {c0, c2};
        std::string labelsExp[] = {"aaa", "ccc"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting everything, once, in-order") {
        numColExp = 3;
        sel = genGivenVals<DTSel>(numColExp, {0, 1, 2});
        std::vector<Structure *> colMatsExp = {c0, c1, c2};
        std::string labelsExp[] = {"aaa", "bbb", "ccc"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting everything, once, permuted") {
        numColExp = 3;
        sel = genGivenVals<DTSel>(numColExp, {2, 0, 1});
        std::vector<Structure *> colMatsExp = {c2, c0, c1};
        std::string labelsExp[] = {"ccc", "aaa", "bbb"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting some, repeated") {
        numColExp = 8;
        sel = genGivenVals<DTSel>(numColExp, {1, 2, 2, 0, 1, 0, 1, 2});
        std::vector<Structure *> colMatsExp = {c1, c2, c2, c0, c1, c0, c1, c2};
        std::string labelsExp[] = {"bbb", "ccc", "col_2", "aaa", "col_4", "col_5", "col_6", "col_7"};
        exp = DataObjectFactory::create<Frame>(colMatsExp, labelsExp);
        checkExtractCol(res, arg, sel, exp);
    }

    DataObjectFactory::destroy(c0, c1, c2, arg, sel);
}

TEMPLATE_TEST_CASE(TEST_NAME("Frame error handling"), TAG_KERNELS, int64_t) { // NOLINT(cert-err58-cpp)
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
    DTSel* sel{};

    SECTION("selecting out of bounds, negative") {
        numColExp = 2;
        sel = genGivenVals<DTSel>(numColExp, {-1, 2});
        checkExtractColThrow(res, arg, sel);
    }
    SECTION("selecting out of bounds, too high") {
        numColExp = 2;
        sel = genGivenVals<DTSel>(numColExp, {1, 3});
        checkExtractColThrow(res, arg, sel);
    }

    DataObjectFactory::destroy(c0, c1, c2, arg, sel);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Dense/Generic Matrix"), TAG_KERNELS, (DATA_TYPES), (int64_t, double)) { // NOLINT(cert-err58-cpp)
    using DT = TestType;
    using VT = typename DT::VT;
    using DTEmpty = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;
    
    DT * arg = genGivenVals<DT>(3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    DT * res{};
    DT * exp{};
    DT * sel{};

    SECTION("selecting nothing") {
        sel = static_cast<DT *>(DataObjectFactory::create<DTEmpty>(0, 1, false));
        exp = genGivenVals<DT>(3, {});
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting some, once, in-order") {
        sel = genGivenVals<DT>(2, {0, 2});
        exp = genGivenVals<DT>(3, {
            1, 3,
            4, 6,
            7, 9
        });
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting everything, once, in-order") {
        sel = genGivenVals<DT>(3, {0, 1, 2});
        exp = genGivenVals<DT>(3, {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        });
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting everything, once, permuted") {
        sel = genGivenVals<DT>(3, {2, 0, 1});
        exp = genGivenVals<DT>(3, {
            3, 1, 2,
            6, 4, 5,
            9, 7, 8
        });
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting some, repeated") {
        sel = genGivenVals<DT>(8, {1, 2, 2, 0, 1, 0, 1, 2});
        exp = genGivenVals<DT>(3, {
            2, 3, 3, 1, 2, 1, 2, 3,
            5, 6, 6, 4, 5, 4, 5, 6,
            8, 9, 9, 7, 8, 7, 8, 9
        });
        checkExtractCol(res, arg, sel, exp);
    }
    SECTION("selecting out of bounds, negative") {
        sel = genGivenVals<DT>(2, {-1, 2});
        checkExtractColThrow(res, arg, sel);
    }
    SECTION("selecting out of bounds, too high") {
        sel = genGivenVals<DT>(2, {1, 3});
        checkExtractColThrow(res, arg, sel);
    }

    DataObjectFactory::destroy(arg, sel);
}