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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/CTable.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>
#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("CTable", TAG_KERNELS, (DenseMatrix, CSRMatrix), (int64_t, int32_t, double)) {
    using DTRes = TestType;
    using VT = typename DTRes::VT;

    DenseMatrix<int64_t> * ys = nullptr;
    DenseMatrix<int64_t> * xs = nullptr;
    VT weight;
    int64_t resNumRows;
    int64_t resNumCols;
    DTRes * exp = nullptr;
    
    SECTION("example 1") {
        ys = genGivenVals<DenseMatrix<int64_t>>(1, {0});
        xs = genGivenVals<DenseMatrix<int64_t>>(1, {0});
        weight = 3;
        resNumRows = -1;
        resNumCols = -1;

        exp = genGivenVals<DTRes>(1, {3});
    }
    SECTION("example 2: automatic shape") {
        ys = genGivenVals<DenseMatrix<int64_t>>(4, {1, 4, 5, 4});
        xs = genGivenVals<DenseMatrix<int64_t>>(4, {2, 3, 1, 3});
        weight = 3;
        resNumRows = -1;
        resNumCols = -1;

        exp = genGivenVals<DTRes>(6, {
            0, 0, 0, 0,
            0, 0, 3, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 6,
            0, 3, 0, 0
        });
    }
    SECTION("example 2: crop #rows") {
        ys = genGivenVals<DenseMatrix<int64_t>>(4, {1, 4, 5, 4});
        xs = genGivenVals<DenseMatrix<int64_t>>(4, {2, 3, 1, 3});
        weight = 3;
        resNumRows = 4;
        resNumCols = -1;

        exp = genGivenVals<DTRes>(4, {
            0, 0, 0, 0,
            0, 0, 3, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        });
    }
    SECTION("example 2: crop #cols") {
        ys = genGivenVals<DenseMatrix<int64_t>>(4, {1, 4, 5, 4});
        xs = genGivenVals<DenseMatrix<int64_t>>(4, {2, 3, 1, 3});
        weight = 3;
        resNumRows = -1;
        resNumCols = 3;

        exp = genGivenVals<DTRes>(6, {
            0, 0, 0,
            0, 0, 3,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 3, 0,
        });
    }
    SECTION("example 2: crop both") {
        ys = genGivenVals<DenseMatrix<int64_t>>(4, {1, 4, 5, 4});
        xs = genGivenVals<DenseMatrix<int64_t>>(4, {2, 3, 1, 3});
        weight = 3;
        resNumRows = 4;
        resNumCols = 3;

        exp = genGivenVals<DTRes>(4, {
            0, 0, 0,
            0, 0, 3,
            0, 0, 0,
            0, 0, 0
        });
    }
    SECTION("example 3: more items than cells") {
        ys = genGivenVals<DenseMatrix<int64_t>>(12, {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2});
        xs = genGivenVals<DenseMatrix<int64_t>>(12, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
        weight = 3;
        resNumRows = -1;
        resNumCols = -1;

        exp = genGivenVals<DTRes>(3, {
            6, 6,
            6, 6,
            6, 6
        });
    }
    
    DTRes * res = nullptr;
    ctable(res, ys, xs, weight, resNumRows, resNumCols, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(ys);
    DataObjectFactory::destroy(xs);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}