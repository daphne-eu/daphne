/*
 * Copyright 2024 The DAPHNE Consortium
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
#include <runtime/local/kernels/Bin.h>
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>
#include <type_traits>
#include <vector>

#include <cstdint>

#define DATA_TYPES DenseMatrix, Matrix

template<class DTRes, class DTArg>
void checkBin(const DTArg * arg, size_t numBins, typename DTArg::VT min, typename DTArg::VT max, const DTRes * exp) {
    DTRes * res = nullptr;
    bin<DTRes, DTArg>(res, arg, numBins, min, max, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template<class DTRes, class DTArg>
void checkBinThrows(const DTArg * arg, size_t numBins, typename DTArg::VT min, typename DTArg::VT max) {
    DTRes * res = nullptr;
    CHECK_THROWS(bin<DTRes, DTArg>(res, arg, numBins, min, max, nullptr));
    if(res != nullptr)
        DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("Bin", TAG_KERNELS, (DATA_TYPES), (double, float, int64_t, uint32_t)) {
    using DTArg = TestType;
    using VTArg = typename DTArg::VT;
    using DTRes = DTArg;
    using DTEmpty = typename std::conditional<
                        std::is_same<TestType, Matrix<VTArg>>::value,
                        DenseMatrix<VTArg>,
                        TestType
                    >::type;
    
    
    DTArg * arg = nullptr;
    DTRes * exp = nullptr;

    // fp spec: nan among normal values

    SECTION("(0x0) arg") {
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 0, false));
        checkBin(arg, 42, 100, 200, exp);
    }
    SECTION("(0xn) arg") {
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(0, 3, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(0, 3, false));
        checkBin(arg, 42, 100, 200, exp);
    }
    SECTION("(mx0) arg") {
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmpty>(3, 0, false));
        exp = static_cast<DTRes *>(DataObjectFactory::create<DTEmpty>(3, 0, false));
        checkBin(arg, 42, 100, 200, exp);
    }
    SECTION("numBins > 1, min < max, wo/ out-of-bins values, 1d") {
        arg = genGivenVals<DTArg>(7, {10, 20, 30, 40, 50, 60, 70});
        exp = genGivenVals<DTRes>(7, { 0,  0,  0,  1,  1,  2,  2});
        checkBin(arg, 3, 10, 70, exp);
    }
    SECTION("numBins > 1, min < max, wo/ out-of-bins values, 2d") {
        arg = genGivenVals<DTArg>(4, {10, 20, 30, 40, 50, 60, 70, 70});
        exp = genGivenVals<DTRes>(4, { 0,  0,  0,  1,  1,  2,  2,  2});
        checkBin(arg, 3, 10, 70, exp);
    }
    SECTION("numBins > 1, min < max, w/ out-of-bins values") {
        arg = genGivenVals<DTArg>(7, {5, 20, 30, 40, 50, 60, 100});
        exp = genGivenVals<DTRes>(7, {0,  0,  0,  1,  1,  2,   2});
        checkBin(arg, 3, 10, 70, exp);
    }
    if constexpr(std::is_floating_point<VTArg>::value) {
        SECTION("numBins > 1, min < max, nan/inf/-inf values") {
            const VTArg inf = std::numeric_limits<VTArg>::infinity();
            const VTArg nan = std::numeric_limits<VTArg>::signaling_NaN();
            arg = genGivenVals<DTArg>(3, {nan, inf, -inf});

            DTRes * res = nullptr;
            bin<DTRes, DTArg>(res, arg, 3, 10, 70, nullptr);
            CHECK(res->getNumRows() == 3);
            CHECK(res->getNumCols() == 1);
            CHECK(std::isnan(res->get(0, 0)));
            CHECK(res->get(1, 0) == VTArg(2));
            CHECK(res->get(2, 0) == VTArg(0));
            DataObjectFactory::destroy(res);
        }
    }
    SECTION("numBins == 1, min == max, wo/ out-of-bounds values") {
        arg = genGivenVals<DTArg>(3, {20, 20, 20});
        exp = genGivenVals<DTRes>(3, { 0,  0,  0});
        checkBin(arg, 1, 20, 20, exp);
    }
    SECTION("numBins == 1, min == max, w/ out-of-bounds values") {
        arg = genGivenVals<DTArg>(3, {10, 20, 30});
        exp = genGivenVals<DTRes>(3, { 0,  0,  0});
        checkBin(arg, 1, 20, 20, exp);
    }
    SECTION("numBins == 1, min < max") {
        arg = genGivenVals<DTArg>(3, {10, 30, 20});
        exp = genGivenVals<DTRes>(3, { 0,  0,  0});
        checkBin(arg, 1, 10, 30, exp);
    }
    SECTION("numBins > 1, min == max") {
        arg = genGivenVals<DTArg>(3, {10, 20, 30});
        checkBinThrows<DTRes>(arg, 3, 20, 20);
    }
    SECTION("numBins <= 0") {
        arg = genGivenVals<DTArg>(1, {150});
        checkBinThrows<DTRes>(arg,  0, 100, 200);
        checkBinThrows<DTRes>(arg, -1, 100, 200);
    }
    SECTION("min > max") {
        arg = genGivenVals<DTArg>(1, {150});
        checkBinThrows<DTRes>(arg, 2, 200, 100);
    }
    if constexpr(std::is_floating_point<VTArg>::value) {
        SECTION("min/max is nan/inf/-inf") {
            const VTArg inf = std::numeric_limits<VTArg>::infinity();
            const VTArg nan = std::numeric_limits<VTArg>::signaling_NaN();
            arg = genGivenVals<DTArg>(1, {150});
            checkBinThrows<DTRes>(arg, 1,  nan,  200);
            checkBinThrows<DTRes>(arg, 1,  inf,  200);
            checkBinThrows<DTRes>(arg, 1, -inf,  200);
            checkBinThrows<DTRes>(arg, 1,  100,  nan);
            checkBinThrows<DTRes>(arg, 1,  100,  inf);
            checkBinThrows<DTRes>(arg, 1,  100, -inf);
        }
    }

    DataObjectFactory::destroy(arg);
    if(exp != nullptr)
        DataObjectFactory::destroy(exp);
}