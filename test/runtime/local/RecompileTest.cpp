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
            func.func @main(%arg1: !daphne.Matrix<10x10xf64>, %arg2: !daphne.Matrix<10x10xf64>) -> (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) {
                %20 = "daphne.constant"() {value = 2.0 : f64} : () -> f64
                %21 = "daphne.ewPow"(%arg1, %20) : (!daphne.Matrix<10x10xf64>, f64) -> !daphne.Matrix<10x10xf64>
                %22 = "daphne.ewPow"(%arg2, %20) : (!daphne.Matrix<10x10xf64>, f64) -> !daphne.Matrix<10x10xf64>
                %23 = "daphne.ewAdd"(%arg1, %21) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
                %24 = "daphne.ewAdd"(%arg2, %22) : (!daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>) -> !daphne.Matrix<10x10xf64>
                func.return %23, %24 : !daphne.Matrix<10x10xf64>, !daphne.Matrix<10x10xf64>
            }
        }
    )";

    // Input matrices
    auto *arg1 = genGivenVals<DTArg>(10, {
        0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0,
        3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0,
        0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0,
        3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0,
        0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0,
        3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0,
        0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0
    });

    auto *arg2 = genGivenVals<DTArg>(10, {
        5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0,
        6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 11.0,
        7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0,
        5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0,
        6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 11.0,
        7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0,
        5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0,
        6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 11.0,
        7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0,
        5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0
    });
    
    const DTArg *args[] = {arg1, arg2};

    // Expected output matrices
    DTRes* expected1 = genGivenVals<DTRes>(10, {
        0.0, 2.0, 6.0, 2.0, 6.0, 12.0, 12.0, 20.0, 30.0, 42.0,
        2.0, 6.0, 12.0, 6.0, 12.0, 20.0, 20.0, 30.0, 42.0, 56.0,
        12.0, 20.0, 30.0, 20.0, 30.0, 42.0, 42.0, 56.0, 72.0, 90.0,
        0.0, 2.0, 6.0, 2.0, 6.0, 12.0, 12.0, 20.0, 30.0, 42.0,
        2.0, 6.0, 12.0, 6.0, 12.0, 20.0, 20.0, 30.0, 42.0, 56.0,
        12.0, 20.0, 30.0, 20.0, 30.0, 42.0, 42.0, 56.0, 72.0, 90.0,
        0.0, 2.0, 6.0, 2.0, 6.0, 12.0, 12.0, 20.0, 30.0, 42.0,
        2.0, 6.0, 12.0, 6.0, 12.0, 20.0, 20.0, 30.0, 42.0, 56.0,
        12.0, 20.0, 30.0, 20.0, 30.0, 42.0, 42.0, 56.0, 72.0, 90.0,
        0.0, 2.0, 6.0, 2.0, 6.0, 12.0, 12.0, 20.0, 30.0, 42.0
    });

    DTRes* expected2 = genGivenVals<DTRes>(10, {
        30.0, 42.0, 56.0, 42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 110.0,
        42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 132.0,
        56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 90.0, 110.0, 132.0, 156.0,
        30.0, 42.0, 56.0, 42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 110.0,
        42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 132.0,
        56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 90.0, 110.0, 132.0, 156.0,
        30.0, 42.0, 56.0, 42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 110.0,
        42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 132.0,
        56.0, 72.0, 90.0, 72.0, 90.0, 110.0, 90.0, 110.0, 132.0, 156.0,
        30.0, 42.0, 56.0, 42.0, 56.0, 72.0, 56.0, 72.0, 90.0, 110.0
    });

    const DTRes *expected[] = {expected1, expected2};

    // Results placeholders
    DTRes *result1 = genGivenVals<DTRes>(10, {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });

    DTRes *result2 = genGivenVals<DTRes>(10, {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });

    DTRes *results[] = {result1, result2};

    // initialize daphne context
    auto dctx = setupContextAndLogger();
    recompile(results, 2,args, 2, mlirCode, dctx.get());

    // Check that results are not null
    REQUIRE(results[0] != nullptr);
    REQUIRE(results[1] != nullptr);

    // Verify results against expected matrices
    CHECK(*results[0] == *expected[0]);
    CHECK(*results[1] == *expected[1]);
    
    // Check types
    CHECK(typeid(*results[0]) == typeid(*expected[0]));
    CHECK(typeid(*results[1]) == typeid(*expected[1]));
    
    // Clean up memory
    DataObjectFactory::destroy(arg1);
    DataObjectFactory::destroy(arg2);
    DataObjectFactory::destroy(expected1);
    DataObjectFactory::destroy(expected2);
    DataObjectFactory::destroy(result1);
    DataObjectFactory::destroy(result2);
}

TEMPLATE_TEST_CASE("recompile", TAG_KERNELS, VALUE_TYPES) {
    checkRecompile<Matrix, TestType, TestType>();
}