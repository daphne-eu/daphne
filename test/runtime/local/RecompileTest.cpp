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

#include "runtime/local/datastructures/ValueTypeCode.h"
#include <cstddef>
#include <run_tests.h>

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datagen/GenGivenVals.h>


#include <runtime/local/kernels/Recompile.h>
#include <runtime/local/kernels/RandMatrix.h>

#include <catch.hpp>
#include <tags.h>

#define DATA_TYPES Matrix
#define VALUE_TYPES double

template <template <typename VT> class DT, class VTarg, class VTres> void checkRecompile() {
    using DTArg = DT<VTarg>;
    using DTRes = DT<VTres>;

    // simple linear mlir program
    const char *mlirCode = R"(
    module{
        func.func @loop_body(%arg1: !daphne.Matrix<10x10xf64>, %arg2: !daphne.Matrix<10x10xf64>, %arg3: si64) -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) {
            %21 = "daphne.ewLog"(%arg1, %arg3) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
            %22 = "daphne.ewAdd"(%arg1, %21) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
            %23 = "daphne.ewLog"(%arg2, %arg3) : (!daphne.Matrix<10x10xf64>, si64) -> !daphne.Matrix<10x10xf64>
            %24 = "daphne.ewAdd"(%arg2, %23) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
            func.return %22, %24 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>
        }
    }
    )";

    // Input matrices
    auto *arg1 = genGivenVals<DTArg>(10, {
        0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0,
        3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0,
        // Fill in the rest to make it 10x10
    });

    auto *arg2 = genGivenVals<DTArg>(10, {
        5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0,
        6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 11.0,
        7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0,
        // Fill in the rest to make it 10x10
    });

    const DTArg *args[] = {arg1, arg2};

    // Expected output matrices
    DTRes *expected1 = genGivenVals<DTRes>(10, {
        5.0, 7.0, 9.0, 7.0, 9.0, 11.0, 10.0, 12.0, 14.0, 16.0,
        7.0, 9.0, 11.0, 9.0, 11.0, 13.0, 12.0, 14.0, 16.0, 18.0,
        10.0, 12.0, 14.0, 12.0, 14.0, 16.0, 15.0, 17.0, 19.0, 21.0,
        // Fill in the rest to make it 10x10
    });

    DTRes *expected2 = genGivenVals<DTRes>(10, {
        // Expected output matrix 2 values based on expected operations
    });

    const DTRes *expected[] = {expected1, expected2};

    // Results placeholders
    DTRes *result1 = nullptr;
    DTRes *result2 = nullptr;
    DTRes *results[] = {result1, result2};

    // initialize daphne context
    auto dctx = setupContextAndLogger();
    recompile(results, args, mlirCode, 2, dctx.get());

    // Check that results are not null
    REQUIRE(results[0] != nullptr);
    REQUIRE(results[1] != nullptr);

    // Verify results against expected matrices
    for (size_t k = 0; k < 2; ++k) {
        for (size_t i = 0; i < expected[k]->getNumRows(); ++i) {
            for (size_t j = 0; j < expected[k]->getNumCols(); ++j) {
                CHECK(results[k]->get(i, j) == Approx(expected[k]->get(i, j)).epsilon(0.001));
            }
        }
    }

    // Check types
    CHECK(typeid(*results[0]) == typeid(*expected[0]));
    CHECK(typeid(*results[1]) == typeid(*expected[1]));

    // Clean up memory
    DataObjectFactory::destroy(arg1);
    DataObjectFactory::destroy(arg2);
    DataObjectFactory::destroy(expected1);
    DataObjectFactory::destroy(expected2);
    DataObjectFactory::destroy(results[0]);
    DataObjectFactory::destroy(results[1]);
}

TEMPLATE_TEST_CASE("recompile", TAG_KERNELS, VALUE_TYPES) {
    checkRecompile<Matrix, TestType, TestType>();
}