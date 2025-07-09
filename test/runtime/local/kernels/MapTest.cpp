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

template <class DTRes, class DTArg>
void checkMap(const DTArg *arg, const DTRes *exp, void *func, const int64_t axis, const bool udfReturnsMatrix) {
    DTRes *res = nullptr;
    map(res, arg, func, axis, udfReturnsMatrix, nullptr);
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

    checkMap(m1, mult3_res1, mult3funcPtr, -1, false);
    checkMap(m2, mult3_res2, mult3funcPtr, -1, false);
    checkMap(m3, mult3_res3, mult3funcPtr, -1, false);

    DataObjectFactory::destroy(m1, m2, m3, mult3_res1, mult3_res2, mult3_res3);
}

template <template <typename VT> class DT, typename VTArg, typename VTRes1, typename... VTResN>
std::enable_if_t<(sizeof...(VTResN) > 0)> checkMult3Map() {
    checkMult3Map<DT, VTArg, VTRes1>();
    checkMult3Map<DT, VTArg, VTResN...>();
}

TEMPLATE_TEST_CASE("Map element-wise dense matrix", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkMult3Map<DenseMatrix, TestType, TYPES>();
    // checkMult3Map<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes> VTRes sumOfFirstAndLastFunc(const DenseMatrix<VTArg> *row) {
    return (static_cast<VTRes>(row->getValues()[0]) + static_cast<VTRes>(row->getValues()[row->getNumCols() - 1]));
}

// Matrix
template <typename VTArg, typename VTRes> VTRes sumOfFirstAndLastFunc(const Matrix<VTArg> *row) {
    return (static_cast<VTRes>(row->getValues()[0]) + static_cast<VTRes>(row->getValues()[row->getNumCols() - 1]));
}

template <template <typename VT> class DT, class VTArg, class VTRes> void checkSumOfFirstAndLastMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *sumRowFuncPtr = reinterpret_cast<void *>((VTRes(*)(const DT<VTArg> *)) & sumOfFirstAndLastFunc<VTArg, VTRes>);

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

    checkMap(m1, sumRow_res1, sumRowFuncPtr, 0, false);
    checkMap(m2, sumRow_res2, sumRowFuncPtr, 0, false);
    checkMap(m3, sumRow_res3, sumRowFuncPtr, 0, false);

    DataObjectFactory::destroy(m1, m2, m3, sumRow_res1, sumRow_res2, sumRow_res3);
}

template <template <typename VT> class DT, typename VTArg, typename VTRes1, typename... VTResN>
std::enable_if_t<(sizeof...(VTResN) > 0)> checkSumOfFirstAndLastMap() {
    checkSumOfFirstAndLastMap<DT, VTArg, VTRes1>();
    checkSumOfFirstAndLastMap<DT, VTArg, VTResN...>();
}

TEMPLATE_TEST_CASE("Map row-wise dense matrix (Matrix -> Scalar)", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkSumOfFirstAndLastMap<DenseMatrix, TestType, TYPES>();
    // checkSumOfFirstAndLastMap<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes> DenseMatrix<VTRes> *appendZero(const DenseMatrix<VTArg> *row) {
    auto numCols = row->getNumCols();
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols + 1, true);
    const VTArg *valuesRow = row->getValues();
    VTRes *valuesRes = res->getValues();
    for (size_t c = 0; c < numCols; c++)
        valuesRes[c] = valuesRow[c];
    return res;
}

// Matrix
template <typename VTArg, typename VTRes> Matrix<VTRes> *appendZero(const Matrix<VTArg> *row) {
    auto numCols = row->getNumCols();
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols + 1, true);
    VTRes *valuesRes = res->getValues();
    for (size_t c = 0; c < numCols; c++)
        valuesRes[c] = row->get(0, c);
    return dynamix_cast<Matrix<VTRes> *>(res);
}

template <template <typename VT> class DT, class VTArg, class VTRes> void checkAppendZeroMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *appendZeroFuncPtr = reinterpret_cast<void *>((DT<VTRes> * (*)(const DT<VTArg> *)) & appendZero<VTArg, VTRes>);

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

    auto appendZero_res1 = genGivenVals<DTRes>(3, {
                                                      0,
                                                      1,
                                                      2,
                                                      0,
                                                      1,
                                                      2,
                                                      3,
                                                      0,
                                                      3,
                                                      4,
                                                      5,
                                                      0,
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

    auto appendZero_res2 = genGivenVals<DTRes>(2, {
                                                      1,
                                                      0,
                                                      2,
                                                      0,
                                                      0,
                                                      3,
                                                      1,
                                                      2,
                                                      0,
                                                      0,
                                                  });

    checkMap(m1, appendZero_res1, appendZeroFuncPtr, 0, true);
    checkMap(m2, appendZero_res2, appendZeroFuncPtr, 0, true);

    DataObjectFactory::destroy(m1, m2, appendZero_res1, appendZero_res2);
}

template <template <typename VT> class DT, typename VTArg, typename VTRes1, typename... VTResN>
std::enable_if_t<(sizeof...(VTResN) > 0)> checkAppendZeroMap() {
    checkAppendZeroMap<DT, VTArg, VTRes1>();
    checkAppendZeroMap<DT, VTArg, VTResN...>();
}

TEMPLATE_TEST_CASE("Map row-wise dense matrix (Matrix -> Matrix)", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkAppendZeroMap<DenseMatrix, TestType, TYPES>();
    // checkAppendZeroMap<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes> DenseMatrix<VTRes> *topThreeFunc(const DenseMatrix<VTArg> *col) {
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(3, 1, false);
    const VTArg *valuesCol = col->getValues();
    VTRes *valuesRes = res->getValues();
    for (size_t r = 0; r < 3; r++) {
        valuesRes[0] = static_cast<VTRes>(valuesCol[0]);
        valuesCol += col->getRowSkip();
        valuesRes += res->getRowSkip();
    }
    return res;
}

// Matrix
template <typename VTArg, typename VTRes> Matrix<VTRes> *topThreeFunc(const Matrix<VTArg> *col) {
    auto res = DataObjectFactory::create<DenseMatrix<VTRes>>(3, 1, false);
    res->prepareAppend();
    for (size_t r = 0; r < 3; r++)
        res->append(r, 0, col->get(r, 0));
    res->finishAppend();
    return dynamic_cast<Matrix<VTRes> *>(res);
}

template <template <typename VT> class DT, class VTArg, class VTRes> void checkTopThreeMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *topThreeFuncPtr = reinterpret_cast<void *>((DT<VTRes> * (*)(const DT<VTArg> *)) & topThreeFunc<VTArg, VTRes>);

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

    checkMap(m1, topThree_res1, topThreeFuncPtr, 1, true);
    checkMap(m2, topThree_res2, topThreeFuncPtr, 1, true);

    DataObjectFactory::destroy(m1, topThree_res1, m2, topThree_res2);
}

template <template <typename VT> class DT, typename VTArg, typename VTRes1, typename... VTResN>
std::enable_if_t<(sizeof...(VTResN) > 0)> checkTopThreeMap() {
    checkTopThreeMap<DT, VTArg, VTRes1>();
    checkTopThreeMap<DT, VTArg, VTResN...>();
}

TEMPLATE_TEST_CASE("Map column-wise dense matrix (Matrix -> Matrix)", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkTopThreeMap<DenseMatrix, TestType, TYPES>();
    // checkTopThreeMap<Matrix, TestType, TYPES>();
}

// DenseMatrix
template <typename VTArg, typename VTRes> VTRes isSumEven(const DenseMatrix<VTArg> *col) {
    int64_t sum = 0;
    const VTArg *valuesCol = col->getValues();
    for (size_t r = 0; r < col->getNumRows(); r++) {
        sum += static_cast<int64_t>(valuesCol[0]);
        valuesCol += col->getRowSkip();
    }
    return static_cast<VTRes>(sum % 2 == 0);
}

// Matrix
template <typename VTArg, typename VTRes> VTRes isSumEven(const Matrix<VTArg> *col) {
    int64_t sum = 0;
    for (size_t r = 0; r < col->getNumRows(); r++)
        sum += static_cast<int64_t>(col->get(r, 0));
    return static_cast<VTRes>(sum % 2 == 0);
}

template <template <typename VT> class DT, class VTArg, class VTRes> void checkIsSumEvenMap() {
    using DTArg = DT<VTArg>;
    using DTRes = DT<VTRes>;

    void *isSumEvenFuncPtr = reinterpret_cast<void *>((VTRes(*)(const DT<VTArg> *)) & isSumEven<VTArg, VTRes>);

    auto m1 = genGivenVals<DTArg>(4, {
                                         0,
                                         1,
                                         2,
                                         3,
                                     });

    auto isSumEven_res1 = genGivenVals<DTRes>(1, {
                                                     1,
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

    auto isSumEven_res2 = genGivenVals<DTRes>(1, {
                                                     0,
                                                     0,
                                                     0,
                                                 });

    checkMap(m1, isSumEven_res1, isSumEvenFuncPtr, 1, false);
    checkMap(m2, isSumEven_res2, isSumEvenFuncPtr, 1, false);

    DataObjectFactory::destroy(m1, isSumEven_res1, m2, isSumEven_res2);
}

template <template <typename VT> class DT, typename VTArg, typename VTRes1, typename... VTResN>
std::enable_if_t<(sizeof...(VTResN) > 0)> checkIsSumEvenMap() {
    checkIsSumEvenMap<DT, VTArg, VTRes1>();
    checkIsSumEvenMap<DT, VTArg, VTResN...>();
}

TEMPLATE_TEST_CASE("Map column-wise dense matrix (Matrix -> Scalar)", TAG_KERNELS, TYPES) {
    // Test all combination of types in TYPES
    checkIsSumEvenMap<DenseMatrix, TestType, TYPES>();
    // checkIsSumEvenMap<Matrix, TestType, TYPES>();
}