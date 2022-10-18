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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CastScaObj.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>
#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("castScaObj, scalar to matrix", TAG_KERNELS, (DenseMatrix), (double, float, int64_t, uint64_t, int32_t, uint32_t)) {
    using DTRes = TestType;
    using VTRes = typename DTRes::VT;
    DTRes* res = nullptr;
    DTRes* exp = nullptr;

    SECTION("int64_t to DenseMatrix<VTRes>") {
        int64_t val = 2;
        exp = genGivenVals<DTRes>(1, {VTRes(val)});
        castScaObj<DTRes, int64_t>(res, val, nullptr);
        CHECK(*res == *exp); 
    }

    SECTION("double to DenseMatrix<VTRes>") {
        double val = 2.2;
        exp = genGivenVals<DTRes>(1, {VTRes(val)});
        castScaObj<DTRes, double>(res, val, nullptr);
        CHECK(*res == *exp);
    }
    DataObjectFactory::destroy(exp); 
    DataObjectFactory::destroy(res);
}

TEMPLATE_TEST_CASE("castScaObj, scalar to frame", TAG_KERNELS, double, float, int64_t, uint64_t, int32_t, uint32_t) {
    using VTRes = TestType;
    Frame* exp = nullptr;
    std::vector<Structure *> cols;
    SECTION("double to Frame[VTRes]") {
        double val = 2.2;

        auto m0 = genGivenVals<DenseMatrix<VTRes>>(1, {static_cast<VTRes>(val)});
        cols = {m0};
        exp = DataObjectFactory::create<Frame>(cols, nullptr);

        Frame* res = nullptr;
        castScaObj<Frame, VTRes>(res, val, nullptr);
        CHECK(*res == *exp);

        DataObjectFactory::destroy(m0, exp, res);
    }

    SECTION("int64_t to Frame[VTRes]") {
        int64_t val = 2;

        auto m0 = genGivenVals<DenseMatrix<VTRes>>(1, {static_cast<VTRes>(val)});
        cols = {m0};
        exp = DataObjectFactory::create<Frame>(cols, nullptr);

        Frame* res = nullptr;
        castScaObj<Frame, VTRes>(res, val, nullptr);
        CHECK(*res == *exp);
        
        DataObjectFactory::destroy(m0, exp, res);
    }
}
