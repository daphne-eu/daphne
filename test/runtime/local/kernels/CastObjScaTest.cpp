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
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CastObjSca.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_TEST_CASE("castObjSca, matrix to scalar, single-element", TAG_KERNELS, double, float, int64_t, uint64_t,
                   int32_t, uint32_t) {
    using VTRes = TestType;

    SECTION("DenseMatrix<int64_t> to VTRes") {
        auto arg = genGivenVals<DenseMatrix<int64_t>>(1, {static_cast<int64_t>(2)});
        VTRes exp = VTRes(2);
        VTRes res = castObjSca<VTRes, DenseMatrix<int64_t>>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(arg);
    }
    SECTION("DenseMatrix<double> to VTRes") {
        auto arg = genGivenVals<DenseMatrix<double>>(1, {static_cast<double>(2.2)});
        VTRes exp = VTRes(2.2);
        VTRes res = castObjSca<VTRes, DenseMatrix<double>>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(arg);
    }
}

TEMPLATE_TEST_CASE("castObjSca, matrix to scalar, non-single-element", TAG_KERNELS, double, int64_t, uint32_t) {
    using VT = TestType;

    DenseMatrix<VT> *arg = nullptr;
    SECTION("zero-element") { arg = DataObjectFactory::create<DenseMatrix<VT>>(0, 0, false); }
    SECTION("multi-element (nx1)") { arg = genGivenVals<DenseMatrix<VT>>(2, {VT(1), VT(2)}); }
    SECTION("multi-element (1xm)") { arg = genGivenVals<DenseMatrix<VT>>(1, {VT(1), VT(2)}); }
    SECTION("multi-element (nxm)") { arg = genGivenVals<DenseMatrix<VT>>(2, {VT(1), VT(2), VT(3), VT(4)}); }
    VT res;
    CHECK_THROWS(res = castObjSca<VT, DenseMatrix<VT>>(arg, nullptr));
    DataObjectFactory::destroy(arg);
}

TEMPLATE_TEST_CASE("castObjSca, frame to scalar, single-element", TAG_KERNELS, double, float, int64_t, uint64_t,
                   int32_t, uint32_t) {
    using VTRes = TestType;

    SECTION("Frame[int64_t] to VTRes") {
        auto argC0 = genGivenVals<DenseMatrix<int64_t>>(1, {static_cast<int64_t>(2)});
        std::vector<Structure *> cols = {argC0};
        auto arg = DataObjectFactory::create<Frame>(cols, nullptr);
        VTRes exp = VTRes(2);
        VTRes res = castObjSca<VTRes, Frame>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(argC0, arg);
    }
    SECTION("Frame[double] to VTRes") {
        auto argC0 = genGivenVals<DenseMatrix<double>>(1, {static_cast<double>(2.2)});
        std::vector<Structure *> cols = {argC0};
        auto arg = DataObjectFactory::create<Frame>(cols, nullptr);
        VTRes exp = VTRes(2.2);
        VTRes res = castObjSca<VTRes, Frame>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(argC0, arg);
    }
}

TEMPLATE_TEST_CASE("castObjSca, frame to scalar, non-single-element", TAG_KERNELS, double, int64_t, uint32_t) {
    using VT = TestType;

    Frame *arg = nullptr;
    DenseMatrix<VT> *argC0 = nullptr;
    SECTION("zero-element") {
        argC0 = DataObjectFactory::create<DenseMatrix<VT>>(0, 1, false);
        std::vector<Structure *> cols = {argC0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);
    }
    SECTION("multi-element (nx1)") {
        argC0 = genGivenVals<DenseMatrix<VT>>(2, {VT(1), VT(2)});
        std::vector<Structure *> cols = {argC0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);
    }
    SECTION("multi-element (1xm)") {
        argC0 = genGivenVals<DenseMatrix<VT>>(1, {VT(1)});
        std::vector<Structure *> cols = {argC0, argC0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);
    }
    SECTION("multi-element (nxm)") {
        argC0 = genGivenVals<DenseMatrix<VT>>(2, {VT(1), VT(2)});
        std::vector<Structure *> cols = {argC0, argC0};
        arg = DataObjectFactory::create<Frame>(cols, nullptr);
    }
    VT res;
    CHECK_THROWS(res = castObjSca<VT, Frame>(arg, nullptr));
    DataObjectFactory::destroy(argC0, arg);
}
