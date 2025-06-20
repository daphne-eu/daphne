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

template <typename VTArg, typename VTRes> VTRes mult3func(VTArg arg) { return static_cast<VTRes>(arg) * 3; }

template <template <typename VT> class DT, class VTArg, class VTRes> void checkMult3Map() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *mult3funcPtr = reinterpret_cast<void *>(&mult3func<VTArg, VTRes>);

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

template <template <typename VT> class DT, typename VTArg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkMult3Map() {
    checkMult3Map<DT, VTArg, VTres1>();
    checkMult3Map<DT, VTArg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map element-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkMult3Map<DenseMatrix, TestType, TYPES>();
    // checkMult3Map<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes>
DenseMatrix<VTRes>* sumOfFirstAndLastFunc(const DenseMatrix<VTArg> *row) {
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, 1, false);
    res->getValues()[0] = (static_cast<VTRes>(row->getValues()[0]) + static_cast<VTRes>(row->getValues()[row->getNumCols() - 1]));
    return res;
}

// // Matrix
// template <typename VTArg, typename VTRes>
// Matrix<VTRes>* sumOfFirstAndLastFunc(const Matrix<VTArg> *row) {
//     auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, 1, false);
//     res->append(0, 0, (static_cast<VTRes>(row->get(0, 0)) + static_cast<VTRes>(row->get(0, row->getNumCols() - 1))));
//     return dynamic_cast<Matrix<VTRes>*>(res);
// }

template <template <typename VT> class DT, class VTArg, class VTRes> void checkSumOfFirstAndLastMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *sumRowFuncPtr = reinterpret_cast<void *>(&sumOfFirstAndLastFunc<VTArg, VTRes>);

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

template <template <typename VT> class DT, typename VTArg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkSumOfFirstAndLastMap() {
    checkSumOfFirstAndLastMap<DT, VTArg, VTres1>();
    checkSumOfFirstAndLastMap<DT, VTArg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map row-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkSumOfFirstAndLastMap<DenseMatrix, TestType, TYPES>();
    // checkSumOfFirstAndLastMap<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes>
DenseMatrix<VTRes>* topThreeFunc(const DenseMatrix<VTArg> *col) {
    const VTArg *valuesCol = col->getValues();
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(3, 1, false);
    VTRes *valuesRes = res->getValues();
    for (size_t r = 0; r < 3; r++) {
        valuesRes[0] = static_cast<VTRes>(valuesCol[0]);
        valuesCol += col->getRowSkip();
        valuesRes += res->getRowSkip();
    }
    return res;
}

// // Matrix
// template <typename VTArg, typename VTRes>
// Matrix<VTRes>* topThreeFunc(const Matrix<VTArg> *col) {
//     auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(3, 1, false);
//     res->prepareAppend();
//     for (size_t r = 0; r < 3; r++)
//         res->append(r, 0, col->get(r, 0));
//     res->finishAppend();
//     return dynamic_cast<Matrix<VTRes>*>(res);
// }

template <template <typename VT> class DT, class VTArg, class VTRes> void checkTopThreeMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *topThreeFuncPtr = reinterpret_cast<void *>(&topThreeFunc<VTArg, VTRes>);

    auto m1 = genGivenVals<DTArg>(4, {
                                         0,
                                         1,
                                         2,
                                         3,
                                     });

    auto topThree_res1 = genGivenVals<DTRes>(3, {
                                                    0,
                                                    1,
                                                    2,
                                             });

    auto m2 = genGivenVals<DTArg>(4, {
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

    auto topThree_res2 = genGivenVals<DTRes>(3, {
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
    checkMap(m2, topThree_res2, topThreeFuncPtr, 1);

    DataObjectFactory::destroy(m1, topThree_res1, m2, topThree_res2);
}

template <template <typename VT> class DT, typename VTArg, typename VTres1, typename... VTresN>
std::enable_if_t<(sizeof...(VTresN) > 0)> checkTopThreeMap() {
    checkTopThreeMap<DT, VTArg, VTres1>();
    checkTopThreeMap<DT, VTArg, VTresN...>();
}

TEMPLATE_TEST_CASE("Map column-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkTopThreeMap<DenseMatrix, TestType, TYPES>();
    // checkTopThreeMap<Matrix, TestType, TYPES>();
}