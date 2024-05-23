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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Recode.h>

#include <tags.h>

#include <catch.hpp>

#include <type_traits>
#include <vector>

#include <cstdint>

template<class DTRes, class DTDict, class DTArg>
void checkRecode(const DTArg * arg, bool orderPreserving, const DTRes * expRes, const DTDict * expDict) {
    DTRes * res = nullptr;
    DTDict * dict = nullptr;
    recode<DTRes, DTDict, DTArg>(res, dict, arg, orderPreserving, nullptr);
    CHECK(*res == *expRes);
    CHECK(*dict == *expDict);
    DataObjectFactory::destroy(res, dict);
}

TEMPLATE_PRODUCT_TEST_CASE("Recode", TAG_KERNELS, (DenseMatrix, Matrix), (double, uint32_t)) {
    using DTArg = TestType;
    using VTArg = typename DTArg::VT;
    using DTRes = typename DTArg::template WithValueType<int64_t>;
    using DTEmptyArg = typename std::conditional<
                        std::is_same<DTArg, Matrix<VTArg>>::value,
                        DenseMatrix<VTArg>,
                        DTArg
                    >::type;
    using DTEmptyRes = typename std::conditional<
                        std::is_same<DTArg, Matrix<VTArg>>::value,
                        DenseMatrix<int64_t>,
                        DTRes
                    >::type;
    
    DTArg * arg = nullptr;
    DTRes * expRes = nullptr;
    DTArg * expDict = nullptr;

    SECTION("empty arg, non-order-preserving recoding") {
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmptyArg>(0, 1, false));
        expRes = static_cast<DTRes *>(DataObjectFactory::create<DTEmptyRes>(0, 1, false));
        expDict = static_cast<DTArg *>(DataObjectFactory::create<DTEmptyArg>(0, 1, false));
        checkRecode(arg, false, expRes, expDict);
    }
    SECTION("empty arg, order-preserving recoding") {
        arg = static_cast<DTArg *>(DataObjectFactory::create<DTEmptyArg>(0, 1, false));
        expRes = static_cast<DTRes *>(DataObjectFactory::create<DTEmptyRes>(0, 1, false));
        expDict = static_cast<DTArg *>(DataObjectFactory::create<DTEmptyArg>(0, 1, false));
        checkRecode(arg, true, expRes, expDict);
    }
    SECTION("non-empty arg, non-order-preserving recoding") {
        arg = genGivenVals<DTArg>(8, {33, 22, 55, 22, 22, 11, 44, 55});
        expRes = genGivenVals<DTRes>(8, {0, 1, 2, 1, 1, 3, 4, 2});
        expDict = genGivenVals<DTArg>(5, {33, 22, 55, 11, 44});
        checkRecode(arg, false, expRes, expDict);
    }
    SECTION("non-empty arg, order-preserving recoding") {
        arg = genGivenVals<DTArg>(8, {33, 22, 55, 22, 22, 11, 44, 55});
        expRes = genGivenVals<DTRes>(8, {2, 1, 4, 1, 1, 0, 3, 4});
        expDict = genGivenVals<DTArg>(5, {11, 22, 33, 44, 55});
        checkRecode(arg, true, expRes, expDict);
    }

    DataObjectFactory::destroy(arg, expRes, expDict);
}