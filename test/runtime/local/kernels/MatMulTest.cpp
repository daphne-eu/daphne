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

#include "run_tests.h"

#include <cstdint>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/MatMul.h>
#include <runtime/local/kernels/SliceCol.h>
#include <runtime/local/kernels/SliceRow.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#define DATA_TYPES DenseMatrix, Matrix
#define VALUE_TYPES float, double, int32_t, int64_t

template <class DT>
void checkMatMul(const DT *lhs, const DT *rhs, const DT *exp, DCTX(dctx), bool transa = false, bool transb = false) {
    DT *res = nullptr;
    matMul<DT, DT, DT>(res, lhs, rhs, transa, transb, dctx);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

template <class DTRes, class DTLhs, class DTRhs>
void checkMatMulMixed(const DTLhs *lhs, const DTRhs *rhs, const DTRes *exp, DCTX(dctx), bool transa = false,
                      bool transb = false) {
    DTRes *res = nullptr;
    matMul<DTRes, DTLhs, DTRhs>(res, lhs, rhs, transa, transb, dctx);
    CHECK(*res == *exp);
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("MatMul", TAG_KERNELS, (CSRMatrix, DATA_TYPES), (VALUE_TYPES)) {
    auto dctx = setupContextAndLogger();

    using DT = TestType;

    auto m0 = genGivenVals<DT>(3, {
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                  });
    auto m1 = genGivenVals<DT>(3, {
                                      1,
                                      2,
                                      3,
                                      3,
                                      1,
                                      2,
                                      2,
                                      3,
                                      1,
                                  });
    auto m2 = genGivenVals<DT>(3, {
                                      13,
                                      13,
                                      10,
                                      10,
                                      13,
                                      13,
                                      13,
                                      10,
                                      13,
                                  });
    auto m3 = genGivenVals<DT>(2, {
                                      1,
                                      0,
                                      3,
                                      0,
                                      0,
                                      0,
                                      2,
                                      0,
                                  });
    auto m4 = genGivenVals<DT>(4, {
                                      0,
                                      1,
                                      2,
                                      0,
                                      1,
                                      1,
                                      0,
                                      0,
                                  });
    auto m5 = genGivenVals<DT>(2, {
                                      3,
                                      4,
                                      2,
                                      2,
                                  });
    auto m6 = genGivenVals<DT>(4, {
                                      1,
                                      0,
                                      0,
                                      0,
                                      3,
                                      2,
                                      0,
                                      0,
                                  });
    auto v0 = genGivenVals<DT>(3, {0, 0, 0});
    auto v1 = genGivenVals<DT>(3, {1, 1, 1});
    auto v2 = genGivenVals<DT>(3, {1, 2, 3});
    auto v3 = genGivenVals<DT>(3, {6, 6, 6});
    auto v4 = genGivenVals<DT>(3, {14, 11, 11});
    auto v5 = genGivenVals<DT>(1, {1, 2, 3});
    auto v6 = genGivenVals<DT>(1, {14});
    auto v7 = genGivenVals<DT>(2, {
                                      1,
                                      1,
                                  });
    auto v8 = genGivenVals<DT>(4, {1, 0, 5, 0});

    checkMatMul(m0, m0, m0, dctx.get());
    checkMatMul(m1, m1, m2, dctx.get());
    checkMatMul(m3, m4, m5, dctx.get());
    checkMatMul(m0, v0, v0, dctx.get());
    checkMatMul(m1, v0, v0, dctx.get());
    checkMatMul(m2, v0, v0, dctx.get());
    checkMatMul(m0, v1, v0, dctx.get());
    checkMatMul(m1, v1, v3, dctx.get());
    checkMatMul(m1, v2, v4, dctx.get());
    checkMatMul(m6, v7, v8, dctx.get());
    checkMatMul(v5, v2, v6, dctx.get());

    DataObjectFactory::destroy(m0, m1, m2, m3, m4, m5, m6, v0, v1, v2, v3, v4, v5, v6, v7, v8);
}

TEMPLATE_TEST_CASE("MatMul Dense x Sparse", TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    using LhsDT = DenseMatrix<VT>;
    using RhsDT = CSRMatrix<VT>;
    using ResDT = DenseMatrix<VT>;

    auto dctx = setupContextAndLogger();

    // clang-format off
    auto lhs = genGivenVals<LhsDT>(3, { 1, 0, 2, 3, 4, 0, });
    auto rhs = genGivenVals<RhsDT>(2, { 0, 5, 0, 0, 6, 0, });
    auto exp = genGivenVals<ResDT>(3, { 0, 5, 0, 0, 28, 0, 0, 20, 0, });
    // clang-format on

    checkMatMulMixed<ResDT, LhsDT, RhsDT>(lhs, rhs, exp, dctx.get());

    DataObjectFactory::destroy(lhs, rhs, exp);
}

TEMPLATE_PRODUCT_TEST_CASE("MatMul Transposed", TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto dctx = setupContextAndLogger();

    auto m0 = genGivenVals<DT>(3, {
                                      1,
                                      2,
                                      3,
                                      3,
                                      1,
                                      2,
                                      2,
                                      3,
                                      1,
                                  });
    auto m1 = genGivenVals<DT>(3, {
                                      13,
                                      10,
                                      13,
                                      13,
                                      13,
                                      10,
                                      10,
                                      13,
                                      13,
                                  });
    auto m2 = genGivenVals<DT>(2, {
                                      1,
                                      0,
                                      3,
                                      0,
                                      0,
                                      0,
                                      2,
                                      0,
                                  });
    auto m3 = genGivenVals<DT>(4, {
                                      0,
                                      1,
                                      2,
                                      0,
                                      1,
                                      1,
                                      0,
                                      0,
                                  });
    auto m4 = genGivenVals<DT>(4, {
                                      0,
                                      2,
                                      1,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      2,
                                      6,
                                      5,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                  });
    auto m5 = genGivenVals<DT>(4, {
                                      1,
                                      0,
                                      1,
                                      0,
                                      0,
                                      4,
                                      2,
                                      0,
                                      1,
                                      2,
                                      2,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                  });
    auto v0 = genGivenVals<DT>(3, {1, 1, 1});
    auto v1 = genGivenVals<DT>(3, {1, 2, 3});
    auto v2 = genGivenVals<DT>(3, {13, 13, 10});
    auto v3 = genGivenVals<DT>(4, {1, 1, 1, 1});
    auto v4 = genGivenVals<DT>(2, {3, 2});
    auto v5 = genGivenVals<DT>(2, {
                                      1,
                                      1,
                                  });
    auto v6 = genGivenVals<DT>(4, {1, 0, 5, 0});
    auto v7 = genGivenVals<DT>(1, {1, 2, 3});
    auto v8 = genGivenVals<DT>(1, {14});

    checkMatMul(m0, m0, m1, dctx.get(), true, true);
    checkMatMul(m2, m3, m4, dctx.get(), true, true);
    checkMatMul(m0, v1, v2, dctx.get(), true);
    checkMatMul(m3, v3, v4, dctx.get(), true);
    checkMatMul(m2, v5, v6, dctx.get(), true);
    checkMatMul(m3, m3, m5, dctx.get(), false, true);
    checkMatMul(v1, v7, v8, dctx.get(), true, true);

    DataObjectFactory::destroy(m0, m1, m2, m3, m4, m5, v0, v1, v2, v3, v4, v5, v6);
}

TEMPLATE_PRODUCT_TEST_CASE("MatMul after slicing", TAG_KERNELS "[new]", (DenseMatrix),
                           (float, double, int32_t, int64_t)) {
    using DT = TestType;
    auto dctx = setupContextAndLogger();
    auto argMatrix = genGivenVals<DT>(4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    DT *resMatrix3x4 = nullptr;
    DT *resMatrix3x3 = nullptr;
    DT *resMatrix3x1 = nullptr;
    DT *resMatrix1x3 = nullptr;
    sliceRow(resMatrix3x4, argMatrix, 0, 3, nullptr);
    sliceCol(resMatrix3x3, resMatrix3x4, 0, 3, nullptr);
    sliceCol(resMatrix3x1, resMatrix3x4, 0, 1, nullptr);
    sliceRow(resMatrix1x3, resMatrix3x3, 0, 1, nullptr);

    auto exp0 = genGivenVals<DT>(3, {
                                        38,
                                        44,
                                        50,
                                        98,
                                        116,
                                        134,
                                        158,
                                        188,
                                        218,
                                    });
    auto exp1 = genGivenVals<DT>(3, {
                                        38,
                                        98,
                                        158,
                                    });
    auto exp2 = genGivenVals<DT>(1, {107});
    auto exp3 = genGivenVals<DT>(1, {38});

    checkMatMul(resMatrix3x3, resMatrix3x3, exp0, dctx.get(), false, false);
    checkMatMul(resMatrix3x3, resMatrix3x1, exp1, dctx.get(), false, false);
    checkMatMul(resMatrix3x1, resMatrix3x1, exp2, dctx.get(), true, false);
    checkMatMul(resMatrix3x1, resMatrix1x3, exp3, dctx.get(), true, true);
    DataObjectFactory::destroy(argMatrix);
    DataObjectFactory::destroy(resMatrix3x3);
}
