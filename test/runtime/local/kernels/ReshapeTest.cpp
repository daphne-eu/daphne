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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/ColBind.h>
#include <runtime/local/kernels/Fill.h>
#include <runtime/local/kernels/Reshape.h>
#include <runtime/local/kernels/RowBind.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix, CSRMatrix
#define VALUE_TYPES double, uint32_t

template <class DT> void checkReshape(const DT *arg, size_t numRows, size_t numCols, const DT *exp) {
    DT *res = nullptr;
    reshape<DT, DT>(res, arg, numRows, numCols, nullptr);
    REQUIRE(res != nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("Reshape (valid)", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTView = typename std::conditional<std::is_same<DT, Matrix<VT>>::value, DenseMatrix<VT>, DT>::type;

    // Choices for whether the argument should be a view into a column segment of a larger matrix. For CSRMatrix such
    // views are not supported.
    std::vector<bool> choiceIsViewColSegment = {false};
    if (!std::is_same<DT, CSRMatrix<VT>>::value)
        choiceIsViewColSegment.push_back(true);
    // Choices for the values.
    std::vector<std::pair<std::string, std::vector<VT>>> choiceNamedVals = {
        // all have 12 elements
        {"all zero data", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"allmost all-zero data", {0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 0}},
        {"allmost all non-zero data", {1, 2, 3, 4, 5, 0, 7, 8, 9, 0, 0, 12}},
        {"all non-zero data", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}};
    // Choices for the shapes.
    std::vector<std::pair<size_t, size_t>> choiceShape = {{1, 12}, {3, 4}, {4, 3}, {12, 1}};

    // Systematically try all combinations of:
    // - whether the argument is a view into a row segment of a larger matrix
    // - whether the argument is a view into a column segment of a larger matrix
    // - the values
    // - the argument's shape
    // - the result's shape
    for (bool isViewRowSegment : {false, true})
        for (bool isViewColSegment : choiceIsViewColSegment)
            for (auto namedVals : choiceNamedVals) {
                auto name = namedVals.first;
                auto vals = namedVals.second;
                for (auto shapeArg : choiceShape) {
                    // Create the argument.
                    DTView *argOrig = genGivenVals<DTView>(shapeArg.first, vals);
                    // If the argument shall be a view into a larger matrix, we concatenate row/column vectors
                    // before and after the argument (and extract a view into this larger matrix later).
                    if (isViewRowSegment) {
                        DTView *tmpRow = nullptr;
                        fill(tmpRow, 99, 1, argOrig->getNumCols(), nullptr);
                        DTView *tmp1 = nullptr;
                        rowBind(tmp1, tmpRow, argOrig, nullptr);
                        DataObjectFactory::destroy(argOrig);
                        argOrig = nullptr;
                        rowBind(argOrig, tmp1, tmpRow, nullptr);
                        DataObjectFactory::destroy(tmp1, tmpRow);
                    }
                    if (isViewColSegment) {
                        DTView *tmpCol = nullptr;
                        fill(tmpCol, 99, argOrig->getNumRows(), 1, nullptr);
                        DTView *tmp1 = nullptr;
                        colBind(tmp1, tmpCol, argOrig, nullptr);
                        DataObjectFactory::destroy(argOrig);
                        argOrig = nullptr;
                        colBind(argOrig, tmp1, tmpCol, nullptr);
                        DataObjectFactory::destroy(tmp1, tmpCol);
                    }
                    DT *arg;
                    size_t numRowsArgOrig = argOrig->getNumRows();
                    size_t numColsArgOrig = argOrig->getNumCols();

                    // If the argument shall be a view into a larger matrix, we extract a view; otherwise, the
                    // argument stays as it is.
                    std::string viewName;
                    if (isViewRowSegment) {
                        if (isViewColSegment) {
                            viewName = "view(row and col segment)";
                            if constexpr (std::is_same<DT, CSRMatrix<VT>>::value)
                                // TODO Support extracting column segments from a CSRMatrix (see #219).
                                throw std::runtime_error(
                                    "CSRMatrix does not support extracting views into column segments");
                            else
                                arg = DataObjectFactory::create<DTView>(argOrig, 1, numRowsArgOrig - 1, 1,
                                                                        numColsArgOrig - 1);
                        } else {
                            viewName = "view(row segment)";
                            if constexpr (std::is_same<DT, CSRMatrix<VT>>::value)
                                arg = DataObjectFactory::create<DTView>(argOrig, 1, numRowsArgOrig - 1);
                            else
                                arg = DataObjectFactory::create<DTView>(argOrig, 1, numRowsArgOrig - 1, 0,
                                                                        numColsArgOrig);
                        }
                        DataObjectFactory::destroy(argOrig);
                    } else {
                        if (isViewColSegment) {
                            viewName = "view(col segment)";
                            if constexpr (std::is_same<DT, CSRMatrix<VT>>::value)
                                throw std::runtime_error("ohoh");
                            else
                                arg = DataObjectFactory::create<DTView>(argOrig, 0, numRowsArgOrig, 1,
                                                                        numColsArgOrig - 1);
                            DataObjectFactory::destroy(argOrig);
                        } else {
                            viewName = "no view";
                            arg = argOrig;
                        }
                    }

                    for (auto shapeRes : choiceShape) {
                        DYNAMIC_SECTION(viewName << ", " << name << ", reshape from " << shapeArg.first << 'x'
                                                 << shapeArg.second << " to " << shapeRes.first << 'x'
                                                 << shapeRes.second) {
                            // Create the expected result.
                            const DT *exp = genGivenVals<DT>(shapeRes.first, vals);
                            // Apply the reshape-kernel.
                            checkReshape(arg, shapeRes.first, shapeRes.second, exp);
                            DataObjectFactory::destroy(exp);
                        }
                    }
                    DataObjectFactory::destroy(arg);
                }
            }
}

TEMPLATE_PRODUCT_TEST_CASE("Reshape (invalid)", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTView = typename std::conditional<std::is_same<DT, Matrix<VT>>::value, DenseMatrix<VT>, DT>::type;

    DT *arg = nullptr;
    size_t numRows;
    size_t numCols;
    SECTION("zero to non-zero shape") {
        if constexpr (std::is_same<DT, DenseMatrix<VT>>::value || std::is_same<DT, Matrix<VT>>::value)
            arg = DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false);
        else if constexpr (std::is_same<DT, CSRMatrix<VT>>::value)
            arg = DataObjectFactory::create<DT>(0, 0, 0, true);
        else
            throw std::runtime_error("unsupported data type");
        numRows = 2;
        numCols = 3;
    }
    SECTION("non-zero to zero shape") {
        arg = genGivenVals<DTView>(2, {1, 2, 3, 4, 5, 6});
        numRows = 0;
        numCols = 0;
    }
    SECTION("invalid non-zero to non-zero shape") {
        arg = genGivenVals<DTView>(2, {1, 2, 3, 4, 5, 6});
        numRows = 3;
        numCols = 5;
    }

    DT *res = nullptr;
    CHECK_THROWS(reshape<DT, DT>(res, arg, numRows, numCols, nullptr));
}

TEMPLATE_PRODUCT_TEST_CASE("Reshape - string specific", TAG_KERNELS, (DenseMatrix, Matrix), (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTView = typename std::conditional<std::is_same<DT, Matrix<VT>>::value, DenseMatrix<VT>, DT>::type;

    std::vector<typename DT::VT> vals = {"ab",      "abcd",     "",          "a",   "abcde", "abcdef",
                                         "abcdefg", "abcdefgh", "abcdefghi", " ab", " ",     "123"};
    DT *arg = genGivenVals<DT>(1, vals); // 1x12

    SECTION("valid reshape 1") {
        const DT *exp = genGivenVals<DT>(12, vals); // 12x1
        checkReshape(arg, 12, 1, exp);
        DataObjectFactory::destroy(exp);
    }
    SECTION("valid reshape 2") {
        const DT *exp = genGivenVals<DT>(3, vals); // 3x4
        checkReshape(arg, 3, 4, exp);
        DataObjectFactory::destroy(exp);
    }
    SECTION("view 1") {
        const DTView *initial = genGivenVals<DTView>(3, vals);                                      // 3x4
        const DT *view = static_cast<DT *>(DataObjectFactory::create<DTView>(initial, 0, 3, 2, 4)); // 3x2
        const DT *exp = genGivenVals<DT>(2, {"", "a", "abcdefg", "abcdefgh", " ", "123"});          // 2x3
        checkReshape(view, 2, 3, exp);

        DataObjectFactory::destroy(exp, initial, view);
    }
    SECTION("view 2") {
        const DTView *initial = genGivenVals<DTView>(2, vals);                                        // 2x6
        const DT *view = static_cast<DT *>(DataObjectFactory::create<DTView>(initial, 1, 2, 0, 6));   // 1x6
        const DT *exp = genGivenVals<DT>(3, {"abcdefg", "abcdefgh", "abcdefghi", " ab", " ", "123"}); // 3x2
        checkReshape(view, 3, 2, exp);

        DataObjectFactory::destroy(exp, initial, view);
    }
    SECTION("view 3") {
        const DTView *initial = genGivenVals<DTView>(2, vals);                                      // 2x6
        const DT *view = static_cast<DT *>(DataObjectFactory::create<DTView>(initial, 1, 2, 0, 4)); // 1x4
        const DT *exp = genGivenVals<DT>(2, {"abcdefg", "abcdefgh", "abcdefghi", " ab"});           // 2x2
        checkReshape(view, 2, 2, exp);

        DataObjectFactory::destroy(exp, initial, view);
    }
    SECTION("invalid reshape") {
        DT *res = nullptr;
        CHECK_THROWS(reshape<DT, DT>(res, arg, 5, 2, nullptr));
    }

    DataObjectFactory::destroy(arg);
}