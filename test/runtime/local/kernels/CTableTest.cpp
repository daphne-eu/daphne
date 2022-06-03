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

TEMPLATE_PRODUCT_TEST_CASE("CTable DenseMatrix", TAG_KERNELS, (DenseMatrix), (int64_t, int32_t, double)) {
    using DT = TestType;

    SECTION("Small") {
        auto m0 = genGivenVals<DT>(4, {2,4,5,4,});
        auto m1 = genGivenVals<DT>(4, {0,3,1,3,});

        auto exp = genGivenVals<DT>(6, {
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 2,
            0, 1, 0, 0,
        });
        
        DT * res = nullptr;
        ctable<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);
        
        DataObjectFactory::destroy(m0);
        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("Larger") {
        auto m0 = genGivenVals<DT>(7, {2,4,5,4,5,1,0,});
        auto m1 = genGivenVals<DT>(7, {0,3,1,3,1,6,3,});

        auto exp = genGivenVals<DT>(6, {
            0, 0, 0, 1, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 1, 
            1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 2, 0, 0, 0,
            0, 2, 0, 0, 0, 0, 0,
        });
        
        DT * res = nullptr;
        ctable<DT, DT, DT>(res, m0, m1, nullptr);
        CHECK(*res == *exp);
        
        DataObjectFactory::destroy(m0);
        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("CTable CSRMatrix", TAG_KERNELS, (CSRMatrix), (int64_t, int32_t, double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    SECTION("Small") {
        auto m0 = genGivenVals<DenseMatrix<VT>>(1, {2});
        auto m1 = genGivenVals<DenseMatrix<VT>>(1, {2});

        auto exp = DataObjectFactory::create<DT>(3, 3, 1, true);
        exp->set(2,2, VT(1));
        DT * res = nullptr;
        ctable<DT, DenseMatrix<VT>, DenseMatrix<VT>>(res, m0, m1, nullptr);
        CHECK(*res == *exp);
        
        DataObjectFactory::destroy(m0);
        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }

    SECTION("Larger") {
        auto m0 = genGivenVals<DenseMatrix<VT>>(4, {2,4,5,4,});
        auto m1 = genGivenVals<DenseMatrix<VT>>(4, {0,3,1,3,});

        auto exp = DataObjectFactory::create<DT>(6, 4, 4, true);
        exp->set(2,0, VT(1));
        exp->set(4,3, VT(2));
        exp->set(5,1, VT(1));

        DT * res = nullptr;
        ctable<DT, DenseMatrix<VT>, DenseMatrix<VT>>(res, m0, m1, nullptr);
        CHECK(*res == *exp);
        
        DataObjectFactory::destroy(m0);
        DataObjectFactory::destroy(m1);
        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(res);
    }
}
