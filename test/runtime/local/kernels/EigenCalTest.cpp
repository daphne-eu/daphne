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
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/EigenCal.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>


template<class DT>
void checkEigenCal(const DT * inMat, const DT * exp1, const DT * exp2) {
    DT * res1 = nullptr;
    DT * res2 = nullptr;
    eigenCal<DT,DT, DT>(res1, res2, inMat, nullptr);  
    CHECK(checkEqApprox(res1, exp1, 1e-2, nullptr));
    CHECK(checkEqApprox(res2, exp2, 1e-2, nullptr));
    DataObjectFactory::destroy(res1);
    DataObjectFactory::destroy(res2);
}


TEMPLATE_PRODUCT_TEST_CASE("EigenCal", TAG_KERNELS, (DenseMatrix), (double, float)) {
    using DT=TestType;
      auto m0 = genGivenVals<DT>(3, {
        504, 360, 180,	
        360, 360, 0,
        180, 0,	720       
    });

        auto m1 = genGivenVals<DT>(3, {
       -0.648, 0.655, -0.385,

        0.741, 0.429, -0.516,
        0.172, 0.621, 0.764
    });
    auto v0 = genGivenVals<DT>(3, {
        44.819,
        910.07,
        629.11       
    });
    
    checkEigenCal(m0, v0, m1);
    DataObjectFactory::destroy(m0, m1,  v0);
}

