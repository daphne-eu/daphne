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

    DenseMatrix<VT> * m0 = nullptr;
    DenseMatrix<VT> * m1 = nullptr;
    DTRes * exp = nullptr;
    
    SECTION("Very small") {
        m0 = genGivenVals<DenseMatrix<VT>>(1, {2});
        m1 = genGivenVals<DenseMatrix<VT>>(1, {2});

        exp = genGivenVals<DTRes>(3, {
            0, 0, 0,
            0, 0, 0,
            0, 0, 1,
        });
    }
    SECTION("Small1") {
        m0 = genGivenVals<DenseMatrix<VT>>(4, {2,4,5,4,});
        m1 = genGivenVals<DenseMatrix<VT>>(4, {0,3,1,3,});

        exp = genGivenVals<DTRes>(6, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 2,
            0, 1, 0, 0,
        });
    }
    SECTION("Small2") {
        m0 = genGivenVals<DenseMatrix<VT>>(4, {1,2,3,4,});
        m1 = genGivenVals<DenseMatrix<VT>>(4, {3,2,1,4,});

        exp = genGivenVals<DTRes>(5, {
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 1,
        });
    }
    SECTION("Larger") {
        m0 = genGivenVals<DenseMatrix<VT>>(7, {2,4,5,4,5,1,0,});
        m1 = genGivenVals<DenseMatrix<VT>>(7, {0,3,1,3,1,6,3,});

        exp = genGivenVals<DTRes>(6, {
            0, 0, 0, 1, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 1, 
            1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 2, 0, 0, 0,
            0, 2, 0, 0, 0, 0, 0,
        });
    }
    
    DTRes * res = nullptr;
    ctable(res, m0, m1, nullptr);
    CHECK(*res == *exp);

    DataObjectFactory::destroy(m0);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(exp);
    DataObjectFactory::destroy(res);
}