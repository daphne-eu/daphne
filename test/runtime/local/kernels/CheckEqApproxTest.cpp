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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/kernels/CheckEqApprox.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

#include <cstdint>

// TODO Extend tests to integral value types, they should be handled
// gracefully, too.

#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES float, double

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, original matrices", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {
        0, 0, 1, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6, 7,
        0, 8, 0, 0, 9, 0,
    };
    std::vector<typename DT::VT> vals2 = { 
        0, 0, 1+1e-7, 0, 2, 0,
        0, 0, 0, 0, 0, 0,
        3, 4, 5, 0, 6+1e-7, 7,
        0, 8, 0, 0, 9, 0,
    };
    auto m1 = genGivenVals<DT>(4, vals);
    SECTION("same inst") {
        CHECK(checkEqApprox(m1, m1, 1e-5, nullptr));
    }
    SECTION("diff inst, same size, same cont") {
        auto m2 = genGivenVals<DT>(4, vals);
        CHECK(checkEqApprox(m1, m2, 1e-5, nullptr));
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, diff size, same cont") {
        auto m2 = genGivenVals<DT>(6, vals);
        CHECK_FALSE(checkEqApprox(m1, m2, 1e-5, nullptr));
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, same size, accepted difference default EPS") {
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK(checkEqApprox(m1, m2, 1e-5, nullptr));
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, same size, accepted difference defined EPS") {
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK(checkEqApprox<DT>(m1, m2, 0.01, nullptr));
        DataObjectFactory::destroy(m2);
    }
    SECTION("diff inst, same size, unaccepted difference defined EPS"){
        auto m2 = genGivenVals<DT>(4, vals2);
        CHECK_FALSE(checkEqApprox<DT>(m1, m2, 1e-13, nullptr));
        DataObjectFactory::destroy(m2);
    }
    
    DataObjectFactory::destroy(m1);
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, views on matrices", TAG_KERNELS, (DenseMatrix, Matrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    using DTGen = typename std::conditional<
                        std::is_same<DT, Matrix<VT>>::value,
                        DenseMatrix<VT>,
                        DT
                    >::type;
    
    std::vector<VT> vals = {
        1, 2, 2, 2, 0, 0,
        3, 4, 4, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };
    std::vector<VT> vals2 = { 
        1+1e-3, 2, 2, 2, 0, 0,
        3, 4, 4+1e-3, 4, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 0, 0,
        1, 2, 0, 0, 0, 0,
        3, 4, 0, 0, 1, 2,
    };    

    auto orig1 = genGivenVals<DTGen>(6, vals);
    auto orig2 = genGivenVals<DTGen>(6, vals2); 
    
    SECTION("same inst") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 0, 2, 0, 2));
        CHECK(checkEqApprox(view1, view1, 1e-5, nullptr));
        DataObjectFactory::destroy(view1);
    }
    SECTION("same view on different equal matrices") {
        auto view1 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig1, 0, 2, 0, 2));
        auto view2 = static_cast<DT *>(DataObjectFactory::create<DTGen>(orig2, 0, 2, 0, 2));
        CHECK(checkEqApprox(view1, view2, 1e-2, nullptr));
        CHECK_FALSE(checkEqApprox(view1, view2, 1e-9, nullptr));
        DataObjectFactory::destroy(view1, view2);
    }
    
    DataObjectFactory::destroy(orig1, orig2);
}

TEMPLATE_PRODUCT_TEST_CASE("CheckEqApprox, frames", TAG_KERNELS, (DenseMatrix), (VALUE_TYPES)) {
    using VTArg = typename TestType::VT;

    const size_t numRows = 4;

    auto c0 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(0.0), VTArg(1.1), VTArg(2.2), VTArg(3.3)});
    auto c1 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(4.4), VTArg(5.5), VTArg(6.6), VTArg(7.7)});
    auto c2 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(8.8), VTArg(9.9), VTArg(1.0), VTArg(2.0)});
    auto c3 = genGivenVals<DenseMatrix<VTArg>>(numRows, {VTArg(8.801), VTArg(9.901), VTArg(1.001), VTArg(2.001)});
   
    std::vector<Structure *> cols1 = {c0, c1, c2};
    std::vector<Structure *> cols2 = {c0, c1, c3};
    auto frame1 = DataObjectFactory::create<Frame>(cols1, nullptr);
    auto frame2 = DataObjectFactory::create<Frame>(cols2, nullptr);
    
    CHECK(checkEqApprox(frame1, frame1, 0.00001, nullptr));
    CHECK(checkEqApprox(frame1, frame2, 0.01, nullptr));
    CHECK_FALSE(checkEqApprox(frame1, frame2, 0.000000001, nullptr));

    DataObjectFactory::destroy(frame1);
    DataObjectFactory::destroy(frame2);
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
    DataObjectFactory::destroy(c2);
    DataObjectFactory::destroy(c3);
}