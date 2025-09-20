/*
 * Copyright 2024 The DAPHNE Consortium
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
#include <runtime/local/kernels/Fill.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix, CSRMatrix
#define VALUE_TYPES int64_t, double, uint32_t

TEMPLATE_PRODUCT_TEST_CASE("Fill", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    // Consider different matrix shapes.
    std::vector<std::pair<size_t, size_t>> choiceShape = {
        // empty matrices
        {0, 0},
        {0, 1},
        {0, 5},
        {1, 0},
        {5, 0},
        // non-empty matrices
        {1, 1},      // point
        {1, 5},      // row (tiny)
        {1, 1000},   // row (not tiny)
        {5, 1},      // column (tiny)
        {1000, 1},   // column (not tiny)
        {3, 4},      // rectangle (tiny)
        {1000, 1000} // rectangle (not tiny)
    };
    // Consider different fill values. We focus on zero and one non-zero value here, because these make a difference for
    // sparse matrices; for dense matrices, the fill value should not matter much.
    std::vector<VT> choiceArg = {/* zero */ VT(0), /* non-zero */ VT(2.5)};

    // Systematically try all combinations of matrix shapes and values.
    for (auto shape : choiceShape) {
        for (VT arg : choiceArg) {
            DYNAMIC_SECTION("fill " << shape.first << 'x' << shape.second << " matrix with value " << arg) {
                // Create the expected result.
                DT *exp = nullptr;
                if (shape.first == 0 || shape.second == 0) {
                    // Use the right constructor depending on the data type.
                    if constexpr (std::is_same<DT, DenseMatrix<VT>>::value || std::is_same<DT, Matrix<VT>>::value)
                        exp = DataObjectFactory::create<DenseMatrix<VT>>(shape.first, shape.second, true);
                    else if constexpr (std::is_same<DT, CSRMatrix<VT>>::value)
                        exp = DataObjectFactory::create<CSRMatrix<VT>>(shape.first, shape.second, 0, true);
                    else
                        throw std::runtime_error("unexpected data type");
                } else
                    exp = genGivenVals<DT>(shape.first, std::vector<VT>(shape.first * shape.second, arg));

                // Apply the fill-kernel.
                DT *res = nullptr;
                fill(res, arg, shape.first, shape.second, nullptr);

                // Check the result.
                REQUIRE(res != nullptr);
                CHECK(*res == *exp);

                DataObjectFactory::destroy(exp, res);
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Fill-existing", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    VT arg = 123;
    size_t numRows = 4, numCols = 4;
    DT *res = genGivenVals<DT>(2, {VT(1.5), VT(1.5), VT(1.5), VT(1.5)});
    CHECK_THROWS(fill(res, arg, numRows, numCols, nullptr));
}

TEMPLATE_PRODUCT_TEST_CASE("Fill", TAG_KERNELS, (DenseMatrix), (ALL_STRING_VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    size_t numRows = 3;
    size_t numCols = 4;

    SECTION("empty_string") {
        DenseMatrix<VT> *res = nullptr;
        VT arg = VT("");

        auto *exp = genGivenVals<DenseMatrix<VT>>(
            3, {VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT(""), VT("")});

        fill(res, arg, numRows, numCols, nullptr);
        CHECK(*exp == *res);

        DataObjectFactory::destroy(res, exp);
    }

    SECTION("not_empty_string") {
        DenseMatrix<VT> *res = nullptr;
        VT arg = VT("abc");

        auto *exp =
            genGivenVals<DenseMatrix<VT>>(3, {VT("abc"), VT("abc"), VT("abc"), VT("abc"), VT("abc"), VT("abc"),
                                              VT("abc"), VT("abc"), VT("abc"), VT("abc"), VT("abc"), VT("abc")});

        fill(res, arg, numRows, numCols, nullptr);
        CHECK(*exp == *res);

        DataObjectFactory::destroy(res, exp);
    }
}
