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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Reshape.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

template<class DT>
void checkReshape(const DT * arg, size_t numRows, size_t numCols, const DT * exp) {
    DT * res = nullptr;
    reshape<DT, DT>(res, arg, numRows, numCols, nullptr); // w/o template
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("Reshape", TAG_KERNELS, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;
    
    std::vector<typename DT::VT> vals = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    DT * arg = genGivenVals<DT>(1, vals); // 1x12
    
    SECTION("valid reshape 1") {
        const DT * exp = genGivenVals<DT>(12, vals); // 12x1
        checkReshape(arg, 12, 1, exp);
        DataObjectFactory::destroy(exp);
    }
    SECTION("valid reshape 2") {
        const DT * exp = genGivenVals<DT>(3, vals); // 3x4
        checkReshape(arg, 3, 4, exp);
        DataObjectFactory::destroy(exp);
    }
    SECTION("view 1") {
        const DT * initial = genGivenVals<DT>(3, vals); // 3x4
        const DT * view = DataObjectFactory::create<DT>(initial, 0, 3, 2, 4); // 3x2
        const DT * exp = genGivenVals<DT>(2, {2,3,6, 7,10,11}); // 2x3
        checkReshape(view, 2, 3, exp);

        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(initial);
        DataObjectFactory::destroy(view);
    }
    SECTION("view 2") {
        const DT * initial = genGivenVals<DT>(2, vals); // 2x6
        const DT * view = DataObjectFactory::create<DT>(initial, 1, 2, 0, 6); // 1x6
        const DT * exp = genGivenVals<DT>(3, {6,7 ,8,9, 10,11}); // 3x2
        checkReshape(view, 3, 2, exp);

        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(initial);
        DataObjectFactory::destroy(view);
    }
    SECTION("view 3") {
        const DT * initial = genGivenVals<DT>(2, vals); // 2x6
        const DT * view = DataObjectFactory::create<DT>(initial, 1, 2, 0, 4); // 1x4
        const DT * exp = genGivenVals<DT>(2, {6,7 ,8,9}); // 2x2
        checkReshape(view, 2, 2, exp);

        DataObjectFactory::destroy(exp);
        DataObjectFactory::destroy(initial);
        DataObjectFactory::destroy(view);
    }
    SECTION("invalid reshape") {
        DT * res = nullptr;
        CHECK_THROWS(reshape<DT, DT>(res, arg, 5, 2, nullptr));
    }

    DataObjectFactory::destroy(arg);
}