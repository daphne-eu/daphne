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

#include "runtime/local/datastructures/ContiguousTensor.h"
#include "runtime/local/datastructures/Tensor.h"
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CastObjSca.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>
#include <catch.hpp>
#include <vector>
#include <cstdint>

TEMPLATE_TEST_CASE("castObjSca, matrix to scalar", TAG_KERNELS, double, float, int64_t, uint64_t, int32_t, uint32_t) {
    using VTRes = TestType;

    VTRes res = VTRes(0);
    SECTION("DenseMatrix<int64_t> to VTRes") {
        VTRes exp = 2;
        auto m0 = genGivenVals<DenseMatrix<int64_t>>(1, {static_cast<int64_t>(exp)});
        res = castObjSca<VTRes, DenseMatrix<int64_t>>(m0, nullptr);
        CHECK(res == exp); 
        DataObjectFactory::destroy(m0);
    }

    SECTION("DenseMatrix<double> to VTRes") {
        VTRes exp = static_cast<VTRes>(2.2);
        auto m0 = genGivenVals<DenseMatrix<double>>(1, {static_cast<double>(exp)});
        res = castObjSca<VTRes, DenseMatrix<double>>(m0, nullptr);
        CHECK(res == exp); 
        DataObjectFactory::destroy(m0);
    }
}

TEMPLATE_TEST_CASE("castObjSca, frame to scalar", TAG_KERNELS, double, float, int64_t, uint64_t, int32_t, uint32_t) {
    using VTRes = TestType;

    Frame* arg = nullptr;
    SECTION("Frame[double] to VTRes") {
        VTRes exp = static_cast<VTRes>(2.2);

        auto m0 = genGivenVals<DenseMatrix<double>>(1, {static_cast<double>(exp)});
        std::vector<Structure *> cols = {m0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);

        VTRes res = castObjSca<VTRes, Frame>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(m0);
    }

    SECTION("Frame[int64_t] to VTRes") {
        VTRes exp = 2;

        auto m0 = genGivenVals<DenseMatrix<int64_t>>(1, {static_cast<int64_t>(exp)});
        std::vector<Structure *> cols = {m0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);

        VTRes res = castObjSca<VTRes, Frame>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(m0);
    }
    DataObjectFactory::destroy(arg);
}


TEMPLATE_TEST_CASE("castObjSca, ContiguousTensor to scalar", TAG_KERNELS, double, float, int64_t, uint64_t, int32_t, uint32_t) {
    using VTRes = TestType;

    using VTArg1 = double;
    using VTArg2 = uint32_t;

    auto t1 = DataObjectFactory::create<ContiguousTensor<VTArg1>>(std::vector<size_t>({}), InitCode::NONE);
    t1->data[0] = static_cast<VTArg1>(42.42);

    VTRes res1 = castObjSca<VTRes, ContiguousTensor<VTArg1>>(t1, nullptr);

    REQUIRE(res1 == static_cast<VTRes>(static_cast<VTArg1>(42.42)));
    
    auto t2 = DataObjectFactory::create<ContiguousTensor<VTArg2>>(std::vector<size_t>({}), InitCode::NONE);
    t2->data[0] = static_cast<VTArg2>(42.42);

    VTRes res2 = castObjSca<VTRes, ContiguousTensor<VTArg2>>(t2, nullptr);

    REQUIRE(res2 == static_cast<VTRes>(static_cast<VTArg2>(42.42)));
}

TEMPLATE_TEST_CASE("castObjSca, ChunkedTensor to scalar", TAG_KERNELS, double, float, int64_t, uint64_t, int32_t, uint32_t) {
    using VTRes = TestType;

    using VTArg1 = double;
    using VTArg2 = uint32_t;

    auto t1 = DataObjectFactory::create<ChunkedTensor<VTArg1>>(std::vector<size_t>({}), std::vector<size_t>({}), InitCode::NONE);
    t1->data[0] = static_cast<VTArg1>(42.42);

    VTRes res1 = castObjSca<VTRes, ChunkedTensor<VTArg1>>(t1, nullptr);

    REQUIRE(res1 == static_cast<VTRes>(static_cast<VTArg1>(42.42)));
    
    auto t2 = DataObjectFactory::create<ChunkedTensor<VTArg2>>(std::vector<size_t>({}), std::vector<size_t>({}), InitCode::NONE);
    t2->data[0] = static_cast<VTArg2>(42.42);

    VTRes res2 = castObjSca<VTRes, ChunkedTensor<VTArg2>>(t2, nullptr);

    REQUIRE(res2 == static_cast<VTRes>(static_cast<VTArg2>(42.42)));
}
