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
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CastObjSca.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

#define MATRIX_DATA_TYPES DenseMatrix, CSRMatrix
#define VALUE_TYPES double, float, int64_t, uint64_t, int32_t, uint32_t, int8_t, uint8_t

TEMPLATE_PRODUCT_TEST_CASE("castObjSca, matrix to scalar, single-element", TAG_KERNELS, (MATRIX_DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    for (int64_t val : {0, 2}) {
        DYNAMIC_SECTION("matrix of int64_t to scalar of given type - " << (val ? "non-" : "") << "zero value") {
            auto arg = genGivenVals<typename DT::WithValueType<int64_t>>(1, {val});
            VT exp = VT(val);
            VT res = castObjSca<VT>(arg, nullptr);
            CHECK(res == exp);
            DataObjectFactory::destroy(arg);
        }
    }
    for (double val : {0.0, 2.2}) {
        DYNAMIC_SECTION("matrix of double to scalar of given type - " << (val ? "non-" : "") << "zero value") {
            auto arg = genGivenVals<typename DT::WithValueType<double>>(1, {val});
            VT exp = VT(val);
            VT res = castObjSca<VT>(arg, nullptr);
            CHECK(res == exp);
            DataObjectFactory::destroy(arg);
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("castObjSca, matrix to scalar, non-single-element", TAG_KERNELS, (MATRIX_DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    DT *arg = nullptr;
    SECTION("zero-element") {
        if constexpr (std::is_same_v<DT, DenseMatrix<VT>>)
            arg = DataObjectFactory::create<DT>(0, 0, false);
        else /* if(std::is_same_v<DT, CSRMatrix<VT>>) */
            arg = DataObjectFactory::create<DT>(0, 0, 0, true);
    }
    SECTION("multi-element (nx1)") { arg = genGivenVals<DT>(2, {VT(1), VT(2)}); }
    SECTION("multi-element (1xm)") { arg = genGivenVals<DT>(1, {VT(1), VT(2)}); }
    SECTION("multi-element (nxm)") { arg = genGivenVals<DT>(2, {VT(1), VT(2), VT(3), VT(4)}); }
    VT res;
    CHECK_THROWS(res = castObjSca<VT>(arg, nullptr));
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

TEMPLATE_TEST_CASE("castObjSca, column to scalar, single-element", TAG_KERNELS, double, float, int64_t, uint64_t,
                   int32_t, uint32_t) {
    using VTRes = TestType;

    SECTION("Column<int64_t> to VTRes") {
        auto arg = genGivenVals<Column<int64_t>>(1, {static_cast<int64_t>(2)});
        VTRes exp = VTRes(2);
        VTRes res = castObjSca<VTRes, Column<int64_t>>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(arg);
    }
    SECTION("Column<double> to VTRes") {
        auto arg = genGivenVals<Column<double>>(1, {static_cast<double>(2.2)});
        VTRes exp = VTRes(2.2);
        VTRes res = castObjSca<VTRes, Column<double>>(arg, nullptr);
        CHECK(res == exp);
        DataObjectFactory::destroy(arg);
    }
}

TEMPLATE_TEST_CASE("castObjSca, column to scalar, non-single-element", TAG_KERNELS, double, int64_t, uint32_t) {
    using VT = TestType;

    Column<VT> *arg = nullptr;
    SECTION("zero-element") { arg = DataObjectFactory::create<Column<VT>>(0, false); }
    SECTION("multi-element (nx1)") { arg = genGivenVals<Column<VT>>(2, {VT(1), VT(2)}); }
    // 1xm column is not possible
    // nxm column is not possible
    VT res;
    CHECK_THROWS(res = castObjSca<VT, Column<VT>>(arg, nullptr));
    DataObjectFactory::destroy(arg);
}
