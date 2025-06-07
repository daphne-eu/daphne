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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/Map.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define TYPES double, float, int64_t, int32_t, int8_t, uint64_t, uint8_t

template <class DTRes, class DTArg> void checkMap(const DTArg *arg, const DTRes *exp, void *func, int64_t axis) {
    DTRes *res = nullptr;
    map(res, arg, func, axis, nullptr);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template <typename VTarg, typename VTres> VTres mult3func(VTarg arg) { return static_cast<VTres>(arg) * 3; }

template <template <typename VT> class DT, class VTarg, class VTres> void checkMult3Map() {
    using DTArg = DT<VTarg>;
    using DTRes = DT<VTres>;

    void *mult3funcPtr = reinterpret_cast<void *>(&mult3func<VTarg, VTres>);

    auto m1 = genGivenVals<DTArg>(3, {
                                         0,
                                         1,
                                         2,
                                         1,
                                         2,
                                         3,
                                         3,
                                         4,
                                         5,
                                     });

    auto mult3_res1 = genGivenVals<DTRes>(3, {
                                                 0,
                                                 3,
                                                 6,
                                                 3,
                                                 6,
                                                 9,
                                                 9,
                                                 12,
                                                 15,
                                             });

    auto m2 = genGivenVals<DTArg>(2, {
                                         1,
                                         0,
                                         2,
                                         0,
                                         3,
                                         1,
                                         2,
                                         0,
                                     });

    auto mult3_res2 = genGivenVals<DTRes>(2, {
                                                 3,
                                                 0,
                                                 6,
                                                 0,
                                                 9,
                                                 3,
                                                 6,
                                                 0,
                                             });

    auto m3 = genGivenVals<DTArg>(3, {1, 5, 3});

    auto mult3_res3 = genGivenVals<DTRes>(3, {3, 15, 9});

    checkMap(m1, mult3_res1, mult3funcPtr, -1);
    checkMap(m2, mult3_res2, mult3funcPtr, -1);
    checkMap(m3, mult3_res3, mult3funcPtr, -1);

    DataObjectFactory::destroy(m1, m2, m3, mult3_res1, mult3_res2, mult3_res3);
}

template <template <typename VT> class DT, typename VTarg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkMult3Map() { // handles multiple template args
    checkMult3Map<DT, VTarg, VTres1>();
    checkMult3Map<DT, VTarg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map element-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkMult3Map<DenseMatrix, TestType, TYPES>();
    checkMult3Map<Matrix, TestType, TYPES>();
}

template <typename VTarg, typename VTres>
VTres sumOfFirstAndLastFunc(DenseMatrix<VTarg> *row) {
    return static_cast<VTres>(row->getValues()[0]) + static_cast<VTres>(row->getValues()[row->getNumCols() - 1]);
}

template <template <typename VT> class DT, class VTarg, class VTres> void checkSumOfFirstAndLastMap() {
    using DTArg = DT<VTarg>;
    using DTRes = DT<VTres>;

    void *sumRowFuncPtr = reinterpret_cast<void *>(&sumOfFirstAndLastFunc<VTarg, VTres>);

    auto m1 = genGivenVals<DTArg>(3, {
                                         0,
                                         1,
                                         2,
                                         1,
                                         2,
                                         3,
                                         3,
                                         4,
                                         5,
                                     });

    auto sumRow_res1 = genGivenVals<DTRes>(3, {
                                                   2,
                                                   4,
                                                   8,
                                               });

    auto m2 = genGivenVals<DTArg>(2, {
                                         1,
                                         0,
                                         2,
                                         0,
                                         3,
                                         1,
                                         2,
                                         0,
                                     });

    auto sumRow_res2 = genGivenVals<DTRes>(2, {1, 3});

    auto m3 = genGivenVals<DTArg>(3, {1, 5, 3});

    auto sumRow_res3 = genGivenVals<DTRes>(3, {2, 10, 6});

    checkMap(m1, sumRow_res1, sumRowFuncPtr, 0);
    checkMap(m2, sumRow_res2, sumRowFuncPtr, 0);
    checkMap(m3, sumRow_res3, sumRowFuncPtr, 0);

    DataObjectFactory::destroy(m1, m2, m3, sumRow_res1, sumRow_res2, sumRow_res3);
}

template <template <typename VT> class DT, typename VTarg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkSumOfFirstAndLastMap() {
    checkSumOfFirstAndLastMap<DT, VTarg, VTres1>();
    checkSumOfFirstAndLastMap<DT, VTarg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map row-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkSumOfFirstAndLastMap<DenseMatrix, TestType, TYPES>();
    checkSumOfFirstAndLastMap<Matrix, TestType, TYPES>();
}

template <typename VTarg, typename VTres>
DenseMatrix<VTres>* topThreeFunc(DenseMatrix<VTarg> *col) {
    return dynamic_cast<DenseMatrix<VTres>*>(col->sliceRow(0, 3));
}

template <template <typename VT> class DT, class VTarg, class VTres> void checkTopThreeMap() {
    using DTArg = DT<VTarg>;
    using DTRes = DT<VTres>;

    void *topThreeFuncPtr = reinterpret_cast<void *>(&topThreeFunc<VTarg, VTres>);

    auto m1 = genGivenVals<DTArg>(4, {
                                         0,
                                         1,
                                         2,
                                         1,
                                         2,
                                         3,
                                         3,
                                         4,
                                         5,
                                         5,
                                         6,
                                         7,
                                     });

    auto topThree_res1 = genGivenVals<DTRes>(3, {
                                                    0,
                                                    1,
                                                    2,
                                                    1,
                                                    2,
                                                    3,
                                                    3,
                                                    4,
                                                    5,
                                             });

    checkMap(m1, topThree_res1, topThreeFuncPtr, 1);

    DataObjectFactory::destroy(m1, topThree_res1);
}

template <template <typename VT> class DT, typename VTarg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkTopThreeMap() {
    checkTopThreeMap<DT, VTarg, VTres1>();
    checkTopThreeMap<DT, VTarg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map column-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkTopThreeMap<DenseMatrix, TestType, TYPES>();
    checkTopThreeMap<Matrix, TestType, TYPES>();
}